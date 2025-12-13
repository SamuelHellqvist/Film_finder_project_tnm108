"""Lightweight script to run the classifier against canned test cases."""

#from .main import run_classifier as run_classifier
from .main import run_classifier  # Import the function
from .test_cases_desc import TEST_CASES
import itertools
import numpy as np


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


def calculate_mrr(results):
    """Calculates Mean Reciprocal Rank (MRR) from evaluation results."""
    reciprocal_ranks = []
    
    for item in results:
        expected_title = item["expected"]
        
        # Check recommendations for the expected title
        found = False
        for rank, rec in enumerate(item["recommendations"], 1):
            if rec.get("title") == expected_title:
                # First match determines the rank for RR
                reciprocal_ranks.append(1.0 / rank)
                found = True
                break
        
        # If the expected title was not in the top_n recommendations, RR is 0
        if not found:
            reciprocal_ranks.append(0.0)

    # MRR is the mean of all reciprocal ranks
    if not reciprocal_ranks:
        return 0.0
        
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def print_evaluation(results):
    """Pretty-print evaluation results."""
    for idx, item in enumerate(results, 1):
        print(f"\n--- Test Case {idx}: expecting '{item['expected']}' ---")
        print(f"Input: {item['input']}")
        for rank, rec in enumerate(item["recommendations"], 1):
            title = rec.get("title", "<missing title>")
            score = rec.get("score", 0.0)
            print(f"{rank:2d}. {title} (score={score:.4f})")


def optimize_weights_grid_search(top_n=3):
    """
    Performs a Grid Search to find optimal weights (w_emb, w_sent, w_key) 
    that maximize the Mean Reciprocal Rank (MRR).
    """
    
    # Define the search space for weights (e.g., from 0.0 to 2.0 in steps of 0.5)
    weight_options = np.arange(0.0, 2, 0.5)
    
    # Initialize optimization variables
    best_mrr = -1.0
    best_weights = {}
    
    # Iterate through all combinations of weights
    # itertools.product creates the full Cartesian product (the grid)
    for w_emb, w_sent, w_key in itertools.product(weight_options, repeat=3):
        
        # Skip the trivial case where all weights are zero
        if w_emb == 0 and w_sent == 0 and w_key == 0:
            continue

        current_weights = {
            "w_emb": w_emb,
            "w_sent": w_sent,
            "w_key": w_key
        }
        
        # 1. Run the ensemble with current weights
        results = run_evaluation(current_weights, top_n=top_n)
        
        # 2. Calculate the performance metric (MRR)
        current_mrr = calculate_mrr(results) 
        
        # 3. Track the best result
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            best_weights = current_weights
            
        # Optional: Print progress
        # print(f"Weights ({w_emb}, {w_sent}, {w_key}): MRR = {current_mrr:.4f}")

    print("\n===========================================")
    print("OPTIMAL ENSEMBLE FOUND VIA GRID SEARCH)")
    print("===========================================")
    print(f"Optimal Weights: {best_weights}")
    print(f"Best MRR (Top {top_n}): {best_mrr:.4f}")
    
    return best_weights, best_mrr


if __name__ == "__main__":
    # The default behavior now runs the optimization to find the best weights
    optimize_weights_grid_search()
    
    # If you wanted to run a single evaluation after optimization:
    # results = run_evaluation(weights={"w_emb": 1.0, "w_sent": 1.0, "w_key": 1.0})
    # print_evaluation(results)