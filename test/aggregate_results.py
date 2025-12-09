#!/usr/bin/env python3
"""
Aggregate multiple rag_vs_baseline_results*.json files and render a stacked
comparison chart (better / tie / worse for each model).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate rag_vs_baseline_results*.json files into a stacked chart."
    )
    parser.add_argument(
        "--results",
        nargs="*",
        default=None,
        help="Explicit JSON result paths. Defaults to all rag_vs_baseline_results*.json under test/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test/rag_vs_baseline_comparison.png"),
        help="Path of the aggregated comparison image.",
    )
    parser.add_argument(
        "--candidate-label",
        default="Candidate wins",
        help="Legend label for candidate wins.",
    )
    parser.add_argument(
        "--baseline-label",
        default="Baseline wins",
        help="Legend label for baseline wins.",
    )
    parser.add_argument(
        "--tie-label",
        default="Tie",
        help="Legend label for ties.",
    )
    return parser.parse_args()


def gather_result_paths(explicit: Sequence[str] | None) -> List[Path]:
    if explicit:
        paths = [Path(p).resolve() for p in explicit]
    else:
        paths = sorted(Path("test").glob("rag_vs_baseline_results*.json"))
    return [p for p in paths if p.exists()]


def load_summary(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    better = summary.get("candidate_wins")
    worse = summary.get("baseline_wins")
    equal = summary.get("ties")

    if better is None or worse is None or equal is None:
        rows = data.get("rows", [])
        counts = {"better": 0, "worse": 0, "equal": 0}
        for row in rows:
            verdict = (row.get("verdict") or "").lower()
            if verdict in counts:
                counts[verdict] += 1
        better = counts["better"]
        worse = counts["worse"]
        equal = counts["equal"]

    label = (
        data.get("chat_model")
        or summary.get("chat_model")
        or path.stem.replace("rag_vs_baseline_results_", "", 1)
    )
    return {
        "label": label,
        "better": int(better),
        "worse": int(worse),
        "equal": int(equal),
    }


def generate_chart(
    model_counts: List[Dict[str, Any]],
    output_path: Path,
    candidate_label: str,
    tie_label: str,
    baseline_label: str,
) -> None:
    if not model_counts:
        raise RuntimeError("No result files found to aggregate.")

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
    ax.barh(y_pos, better_pct, color="#5DADE2", label=candidate_label)
    ax.barh(y_pos, equal_pct, left=better_pct, color="#F4D03F", label=tie_label)
    ax.barh(
        y_pos,
        worse_pct,
        left=[b + e for b, e in zip(better_pct, equal_pct)],
        color="#E74C3C",
        label=baseline_label,
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    paths = gather_result_paths(args.results)
    if not paths:
        raise RuntimeError("No result files found. Provide paths via --results or run evaluations first.")

    summaries = [load_summary(path) for path in paths]
    logging.info("Loaded %s result file(s).", len(summaries))
    generate_chart(
        summaries,
        args.output,
        candidate_label=args.candidate_label,
        tie_label=args.tie_label,
        baseline_label=args.baseline_label,
    )
    logging.info("Comparison image saved to %s", args.output)


if __name__ == "__main__":
    main()
