"""Evaluation helpers for the ensemble recommender."""

import argparse
import itertools
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .main import run_classifier
from .test_cases_desc import TEST_CASES


Weights = Dict[str, float]
Recommendation = Dict[str, float]
EvalResult = Dict[str, object]


def run_evaluation(
    weights: Weights | None = None,
    top_n: int = 3,
    cases: Iterable[Tuple[str, str]] | None = None,
) -> List[EvalResult]:
    """
    Evaluate the classifier on the given test cases.

    Args:
        weights: mapping with keys w_emb, w_sent, w_key.
        top_n: how many recommendations to keep per test case.
        cases: optional iterable of (title, description). Defaults to TEST_CASES.
    """
    weights = weights or {}
    w_emb = float(weights.get("w_emb", 1.0))
    w_sent = float(weights.get("w_sent", 1.0))
    w_key = float(weights.get("w_key", 1.0))

    summary: List[EvalResult] = []
    cases = cases or TEST_CASES

    for expected_title, description in cases:
        recommendations = run_classifier(description, w_emb, w_sent, w_key)
        summary.append(
            {
                "expected": expected_title,
                "input": description,
                "recommendations": recommendations[:top_n],
            }
        )

    return summary


def calculate_mrr(results: List[EvalResult]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) from evaluation results."""
    reciprocal_ranks: List[float] = []

    for item in results:
        expected_title = item["expected"]
        found = False
        for rank, rec in enumerate(item["recommendations"], 1):
            if rec.get("title") == expected_title:
                reciprocal_ranks.append(1.0 / rank)
                found = True
                break
        if not found:
            reciprocal_ranks.append(0.0)

    return float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0


def calculate_hit_rate(results: List[EvalResult]) -> float:
    """Compute hit-rate@top_n (fraction where expected title appears)."""
    if not results:
        return 0.0
    hits = sum(
        1
        for item in results
        if any(rec.get("title") == item["expected"] for rec in item["recommendations"])
    )
    return hits / len(results)


def print_evaluation(results: List[EvalResult]) -> None:
    """Pretty-print evaluation results."""
    for idx, item in enumerate(results, 1):
        print(f"\n--- Test Case {idx}: expecting '{item['expected']}' ---")
        print(f"Input: {item['input']}")
        for rank, rec in enumerate(item["recommendations"], 1):
            title = rec.get("title", "<missing title>")
            score = rec.get("score", 0.0)
            print(f"{rank:2d}. {title} (score={score:.4f})")

# deema kan gärna ta en bild från hit
def optimize_weights_grid_search(
    top_n: int = 3,
    start: float = 0.0,
    stop: float = 1.0,
    step: float = 0.1,
) -> Tuple[Weights, float]:

    weight_options = np.arange(start, stop, step)
    best_mrr = -1.0
    best_weights: Weights = {}

    for w_emb, w_sent, w_key in itertools.product(weight_options, repeat=3):
        if w_emb == 0 and w_sent == 0 and w_key == 0:
            continue

        current_weights = {"w_emb": float(w_emb), "w_sent": float(w_sent), "w_key": float(w_key)}
        results = run_evaluation(current_weights, top_n=top_n)
        current_mrr = calculate_mrr(results)

        if current_mrr > best_mrr:
            best_mrr = current_mrr
            best_weights = current_weights

    # till hit

    print("\n===========================================")
    print("OPTIMAL ENSEMBLE FOUND VIA GRID SEARCH")
    print("===========================================")
    print(f"Optimal Weights: {best_weights}")
    print(f"Best MRR (Top {top_n}): {best_mrr:.4f}")

    return best_weights, best_mrr


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ensemble recommendations.")
    parser.add_argument("--top-n", type=int, default=3, help="Number of recommendations to keep.")
    parser.add_argument("--grid-start", type=float, default=0.0, help="Grid search start (inclusive).")
    parser.add_argument("--grid-stop", type=float, default=1.0, help="Grid search stop (exclusive).")
    parser.add_argument("--grid-step", type=float, default=0.1, help="Grid search step.")
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Only run grid search (skip final evaluation run).",
    )
    args = parser.parse_args()

    # Always run grid search first so we actually discover and print the best weights.
    best_weights, best_mrr = optimize_weights_grid_search(
        args.top_n, args.grid_start, args.grid_stop, args.grid_step
    )
    print(f"\nBest weights found: {best_weights} (MRR@{args.top_n}={best_mrr:.4f})")

    if args.grid_only:
        return

    # Evaluate once using the best weights discovered.
    results = run_evaluation(best_weights, top_n=args.top_n)
    print("\n-- Evaluation with best weights --")
    print_evaluation(results)
    print("\n-- Metrics --")
    print(f"MRR@{args.top_n}: {calculate_mrr(results):.4f}")
    print(f"HitRate@{args.top_n}: {calculate_hit_rate(results):.4f}")


if __name__ == "__main__":
    main()