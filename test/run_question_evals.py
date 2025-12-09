#!/usr/bin/env python3
"""
Batch test the questions in test/question.json, comparing vanilla LLM answers
against the RAG pipeline answers. The script also asks the LLM to judge which
answer is better (accuracy, fluency, relevance) and produces a comparison chart
that can be extended to include additional models in the future.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")  # Headless-friendly backend.
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, SystemMessage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_config import load_settings
from rag_pipeline import (
    build_rag_chain,
    connect_vectorstore,
    create_llm_clients,
)


SIMILARITY_MARGIN = 0.01


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = sum(x * x for x in vec_a) ** 0.5
    norm_b = sum(y * y for y in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _similarity_to_reference(
    reference: str,
    answer: str,
    embeddings,
) -> float:
    if not reference.strip() or not answer.strip():
        return 0.0
    ref_vec = embeddings.embed_query(reference)
    ans_vec = embeddings.embed_query(answer)
    return _cosine_similarity(ref_vec, ans_vec)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", text).strip("-")
    return slug.lower() or "model"


@dataclass
class QuestionItem:
    """Container for each test question."""

    qid: int
    question: str
    reference_answer: str


@dataclass
class QuestionRunResult:
    """Holds generation and evaluation outputs for one question."""

    qid: int
    question: str
    reference_answer: str
    baseline_answer: str
    rag_answer: str
    verdict: str
    baseline_similarity: float
    rag_similarity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RAG vs baseline answers.")
    parser.add_argument(
        "--question-file",
        type=Path,
        default=Path("test/question.json"),
        help="Path to the question JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of questions (useful for smoke tests).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("test/rag_vs_baseline_results.json"),
        help="Where to store the structured comparison results.",
    )
    parser.add_argument(
        "--comparison-image",
        type=Path,
        default=Path("test/rag_vs_baseline_comparison.png"),
        help="Path for the generated comparison chart.",
    )
    parser.add_argument(
        "--candidate-label",
        default="Candidate answer",
        help="Label to show for the candidate approach.",
    )
    parser.add_argument(
        "--baseline-label",
        default="Baseline model",
        help="Label to show for the baseline (non-RAG) answers.",
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default=None,
        help="Override the chat model for this run (defaults to RAG_CHAT_MODEL).",
    )
    parser.add_argument(
        "--chat-models",
        type=str,
        default=None,
        help="Comma-separated list of chat models to evaluate sequentially.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of questions to evaluate in parallel (default: 1).",
    )
    return parser.parse_args()


def load_questions(path: Path) -> List[QuestionItem]:
    if not path.exists():
        raise FileNotFoundError(f"Question file '{path}' not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = [
        QuestionItem(
            qid=item.get("id", idx + 1),
            question=item["question"],
            reference_answer=item.get("answer", ""),
        )
        for idx, item in enumerate(data)
    ]
    return questions


def _coerce_content(result) -> str:
    if hasattr(result, "content"):
        return str(result.content).strip()
    if isinstance(result, dict) and "content" in result:
        return str(result["content"]).strip()
    return str(result).strip()


def answer_without_rag(model, question: str) -> str:
    """Generate a baseline answer directly from the chat model."""
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "你是一名技术问答助手，请基于通用知识准确回答问题。"
                    "避免编造内容，回答需清晰、专业，并尽量简洁。"
                )
            ),
            HumanMessage(content=question),
        ]
    )
    return _coerce_content(response)


def answer_with_rag(chain, question: str) -> str:
    """Generate an answer via the RAG retrieval chain."""
    response = chain.invoke(question)
    return _coerce_content(response)


def save_results(
    results: List[QuestionRunResult],
    path: Path,
    chat_model_name: str,
) -> None:
    summary_counts = tally_verdicts(result.verdict for result in results)
    payload = {
        "chat_model": chat_model_name,
        "summary": {
            "total_questions": len(results),
            "candidate_wins": summary_counts["better"],
            "baseline_wins": summary_counts["worse"],
            "ties": summary_counts["equal"],
        },
        "rows": [
            {
                "id": item.qid,
                "question": item.question,
                "reference_answer": item.reference_answer,
                "baseline_answer": item.baseline_answer,
                "rag_answer": item.rag_answer,
                "verdict": item.verdict,
                "baseline_similarity": round(item.baseline_similarity, 4),
                "rag_similarity": round(item.rag_similarity, 4),
            }
            for item in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def tally_verdicts(verdicts: Iterable[str]) -> Dict[str, int]:
    counts = {"better": 0, "equal": 0, "worse": 0}
    for verdict in verdicts:
        if verdict in counts:
            counts[verdict] += 1
    return counts


def run_model_evaluation(
    settings,
    questions: List[QuestionItem],
    concurrency: int,
) -> List[QuestionRunResult]:
    chat_model, embeddings = create_llm_clients(settings)
    vectorstore = connect_vectorstore(settings, embeddings)
    rag_chain, _ = build_rag_chain(chat_model, vectorstore, settings)

    def _process_question(item: QuestionItem) -> QuestionRunResult:
        logging.info("[%s] Processing question %s: %s", settings.chat_model, item.qid, item.question)
        baseline_answer = answer_without_rag(chat_model, item.question)
        rag_answer = answer_with_rag(rag_chain, item.question)
        reference_text = item.reference_answer or ""
        baseline_similarity = _similarity_to_reference(
            reference_text,
            baseline_answer,
            embeddings,
        )
        rag_similarity = _similarity_to_reference(
            reference_text,
            rag_answer,
            embeddings,
        )
        delta = rag_similarity - baseline_similarity
        if delta > SIMILARITY_MARGIN:
            verdict = "better"
        elif delta < -SIMILARITY_MARGIN:
            verdict = "worse"
        else:
            verdict = "equal"
        logging.info(
            "[%s] Verdict for question %s: %s (baseline=%.3f, rag=%.3f)",
            settings.chat_model,
            item.qid,
            verdict,
            baseline_similarity,
            rag_similarity,
        )
        return QuestionRunResult(
            qid=item.qid,
            question=item.question,
            reference_answer=item.reference_answer,
            baseline_answer=baseline_answer,
            rag_answer=rag_answer,
            verdict=verdict,
            baseline_similarity=baseline_similarity,
            rag_similarity=rag_similarity,
        )

    results: List[QuestionRunResult] = []
    concurrency = max(1, concurrency)
    if concurrency == 1:
        for item in questions:
            results.append(_process_question(item))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_question = {
                executor.submit(_process_question, item): item for item in questions
            }
            for future in as_completed(future_to_question):
                results.append(future.result())
    results.sort(key=lambda item: item.qid)
    return results


def generate_stacked_comparison_image(
    model_counts: List[Dict[str, Any]],
    output_path: Path,
    better_label: str,
    tie_label: str,
    worse_label: str,
) -> None:
    """Render a stacked horizontal bar chart (better/equal/worse per model)."""
    if not model_counts:
        return

    labels = [item["label"] for item in model_counts]
    better = [item["better"] for item in model_counts]
    equal = [item["equal"] for item in model_counts]
    worse = [item["worse"] for item in model_counts]
    totals = [max(b + e + w, 1) for b, e, w in zip(better, equal, worse)]

    better_pct = [b / t * 100 for b, t in zip(better, totals)]
    equal_pct = [e / t * 100 for e, t in zip(equal, totals)]
    worse_pct = [w / t * 100 for w, t in zip(worse, totals)]

    y_pos = list(range(len(labels)))
    height = max(3, len(labels) * 0.6)
    fig, ax = plt.subplots(figsize=(9, height + 1))

    ax.barh(y_pos, better_pct, color="#5DADE2", label=better_label)
    ax.barh(y_pos, equal_pct, left=better_pct, color="#F4D03F", label=tie_label)
    ax.barh(
        y_pos,
        worse_pct,
        left=[b + e for b, e in zip(better_pct, equal_pct)],
        color="#E74C3C",
        label=worse_label,
    )

    for idx, y in enumerate(y_pos):
        segments = [
            (better_pct[idx], better[idx]),
            (equal_pct[idx], equal[idx]),
            (worse_pct[idx], worse[idx]),
        ]
        offset = 0.0
        for pct, count in segments:
            if pct < 5:
                offset += pct
                continue
            ax.text(
                offset + pct / 2,
                y,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                color="#1C2833",
                fontsize=10,
            )
            offset += pct

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of questions (%)")
    ax.set_title("RAG vs Baseline Outcome by Model")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    questions = load_questions(args.question_file)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise RuntimeError("No questions available for evaluation.")

    base_settings = load_settings()
    if args.chat_model:
        base_settings = replace(base_settings, chat_model=args.chat_model)

    if args.chat_models:
        models = [m.strip() for m in args.chat_models.split(",") if m.strip()]
    else:
        models = [base_settings.chat_model]

    if not models:
        raise ValueError("No chat models specified for evaluation.")

    use_suffix = bool(args.chat_models)
    multi_model = len(models) > 1
    model_counts: List[Dict[str, Any]] = []

    for model_name in models:
        run_settings = replace(base_settings, chat_model=model_name)
        results = run_model_evaluation(run_settings, questions, args.concurrency)
        counts = tally_verdicts(result.verdict for result in results)

        if use_suffix:
            slug = _slugify(model_name)
            output_json = args.output_json.with_name(
                f"{args.output_json.stem}_{slug}{args.output_json.suffix}"
            )
        else:
            output_json = args.output_json

        save_results(results, output_json, model_name)
        model_counts.append(
            {
                "label": model_name,
                "better": counts["better"],
                "equal": counts["equal"],
                "worse": counts["worse"],
            }
        )
        logging.info(
            "Model '%s' finished. Results saved to %s.",
            model_name,
            output_json,
        )

    generate_stacked_comparison_image(
        model_counts,
        args.comparison_image,
        better_label=f"{args.candidate_label} wins",
        tie_label="Tie",
        worse_label=f"{args.baseline_label} wins",
    )
    logging.info(
        "Aggregate chart saved to %s covering model(s): %s",
        args.comparison_image,
        ", ".join(models),
    )


if __name__ == "__main__":
    main()
