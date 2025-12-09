"""Lightweight script to run the classifier against canned test cases."""

#from .main import run_classifier as run_classifier
from .main import run_classifier  # Import the function
from .test_cases_desc import TEST_CASES


def run_evaluation(weights=None, top_n=3):
    """
    Evaluate the classifier on the predefined TEST_CASES.

    Args:
        weights: Optional mapping with keys w_emb, w_sent, w_key.
        top_n: How many recommendations to keep per test case.
    """
    weights = weights or {}
    w_emb = weights.get("w_emb", 1.0)
    w_sent = weights.get("w_sent", 1.0)
    w_key = weights.get("w_key", 1.0)

    summary = []

    for expected_title, description in TEST_CASES:
        recommendations = run_classifier(description, w_emb, w_sent, w_key)
        summary.append(
            {
                "expected": expected_title,
                "input": description,
                "recommendations": recommendations[:top_n],
            }
        )

    return summary


def print_evaluation(results):
    """Pretty-print evaluation results."""
    for idx, item in enumerate(results, 1):
        print(f"\n--- Test Case {idx}: expecting '{item['expected']}' ---")
        print(f"Input: {item['input']}")
        for rank, rec in enumerate(item["recommendations"], 1):
            title = rec.get("title", "<missing title>")
            score = rec.get("score", 0.0)
            print(f"{rank:2d}. {title} (score={score:.4f})")


if __name__ == "__main__":
    results = run_evaluation()
    print_evaluation(results)
