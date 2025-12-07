"""
Run an end-to-end evaluation loop for the technical-document RAG pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

from langchain.evaluation import load_evaluator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from document_processing import DocumentIngestor, chunk_documents
from rag_config import load_settings
from rag_pipeline import build_rag_chain, connect_vectorstore, create_llm_clients


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of auto-generated questions to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path for saving the evaluation JSON report.",
    )
    return parser.parse_args()


def _extract_json(payload: str) -> Dict[str, str]:
    """Best-effort JSON parsing that tolerates stray text around the object."""
    payload = payload.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(payload[start : end + 1])


def generate_eval_pairs(
    llm: BaseChatModel, documents, total: int
) -> List[Dict[str, str]]:
    prompt = ChatPromptTemplate.from_template(
        (
            "作为技术文档评测数据构造器，请根据给定片段生成一个问答对。\n"
            "仅使用片段中的信息，问题应考察关键技术细节，答案要简洁准确。\n"
            "将结果以JSON格式输出，例如: "
            '{{"question": "……", "answer": "……"}}。\n'
            "片段:\n{chunk}"
        )
    )
    candidates = list(documents)
    random.shuffle(candidates)
    qa_pairs: List[Dict[str, str]] = []

    for document in candidates:
        if len(qa_pairs) >= total:
            break
        snippet = document.page_content[:1500]
        response = llm.invoke(prompt.format(chunk=snippet))
        try:
            data = _extract_json(response.content)
            if data.get("question") and data.get("answer"):
                qa_pairs.append(
                    {
                        "question": data["question"].strip(),
                        "answer": data["answer"].strip(),
                        "source": document.metadata.get("source", "unknown"),
                    }
                )
        except Exception as exc:
            logger.warning("Failed to parse eval pair: %s", exc)
            continue

    if len(qa_pairs) < total:
        logger.warning(
            "Only generated %s/%s evaluation pairs. "
            "Consider adding more source material.",
            len(qa_pairs),
            total,
        )
    return qa_pairs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    settings = load_settings()
    samples = args.samples or settings.evaluation_samples
    output_path = Path(args.output or settings.evaluation_output_path)

    ingestor = DocumentIngestor(settings.input_dir, settings.url_manifest)
    raw_docs = ingestor.load()
    chunks = chunk_documents(
        raw_docs,
        settings.chunk_size,
        settings.chunk_overlap,
        min_chunk_chars=settings.min_chunk_chars,
    )

    model, embeddings = create_llm_clients(settings)
    vectorstore = connect_vectorstore(settings, embeddings)
    chain, _ = build_rag_chain(model, vectorstore, settings)

    qa_pairs = generate_eval_pairs(model, chunks, samples)
    if not qa_pairs:
        raise RuntimeError("Unable to generate evaluation questions.")

    evaluator = load_evaluator("context_qa", llm=model)
    results = []
    scores = []

    for pair in qa_pairs:
        prediction = chain.invoke(pair["question"])
        prediction_text = prediction.content if hasattr(prediction, "content") else str(prediction)
        eval_result = evaluator.evaluate_strings(
            prediction=prediction_text,
            reference=pair["answer"],
            input=pair["question"],
        )
        score = eval_result.get("score")
        if isinstance(score, (int, float)):
            scores.append(score)
        results.append(
            {
                "question": pair["question"],
                "reference_answer": pair["answer"],
                "rag_answer": prediction_text,
                "source": pair["source"],
                "evaluation": eval_result,
            }
        )

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_score = sum(scores) / len(scores) if scores else 0
    logger.info("Saved evaluation report to %s", output_path)
    logger.info("Average evaluator score: %.2f (%s samples)", avg_score, len(scores))


if __name__ == "__main__":
    main()
