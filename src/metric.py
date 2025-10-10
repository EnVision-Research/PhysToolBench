import os
import json
import csv
import re
from collections import defaultdict

def normalize_answer(ans):
    """Convert answer string like '1, 3' to a set of stripped strings."""
    if not isinstance(ans, str):
        ans = str(ans)
    return set(s.strip() for s in ans.split(',') if s.strip())

def compute_scores(model_name):
    """Computes and prints accuracy for a given model, categorized by difficulty."""
    result_dir = os.path.join(os.path.dirname(__file__), '../results', model_name)
    if not os.path.exists(result_dir):
        print(f"Result directory not found: {result_dir}")
        return

    files = [f for f in os.listdir(result_dir) if f.endswith('.json')]
    if not files:
        print(f"No result files found in {result_dir}")
        return

    # Group results by difficulty
    results_by_difficulty = defaultdict(list)
    for fname in files:
        fpath = os.path.join(result_dir, fname)
        with open(fpath, 'r') as f:
            try:
                data = json.load(f)
                difficulty = data.get('difficulty', 'Unknown')
                results_by_difficulty[difficulty].append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {fname}")

    print(f"Processing results for model: {model_name}")
    
    # Prepare CSV file
    csv_file_path = os.path.join(result_dir, 'accuracy.csv')
    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model Name', 'Difficulty', 'Metric', 'Accuracy', 'Correct', 'Total'])

    total_correct_cot = 0
    total_correct_no_cot = 0
    total_questions = 0

    difficulties = sorted(results_by_difficulty.keys())

    for difficulty in difficulties:
        difficulty_results = results_by_difficulty[difficulty]
        total = len(difficulty_results)
        correct_cot = 0
        correct_no_cot = 0

        for data in difficulty_results:
            pred_cot = normalize_answer(data.get('cot_answer', ''))
            pred_no_cot = normalize_answer(data.get('no_cot_answer', ''))
            gt = normalize_answer(data.get('correct_tool', ''))

            if pred_cot == gt:
                correct_cot += 1
            if pred_no_cot == gt:
                correct_no_cot += 1
        
        total_questions += total
        total_correct_cot += correct_cot
        total_correct_no_cot += correct_no_cot

        accuracy_cot = correct_cot / total if total > 0 else 0.0
        accuracy_no_cot = correct_no_cot / total if total > 0 else 0.0
        
        print(f"\nDifficulty: {difficulty}")
        print(f"  Accuracy (w/ reasoning): {accuracy_cot:.4f} ({correct_cot}/{total})")
        print(f"  Accuracy (w/o reasoning): {accuracy_no_cot:.4f} ({correct_no_cot}/{total})")

        # Write to CSV
        with open(csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_name, difficulty, 'w/ reasoning', f"{accuracy_cot:.4f}", correct_cot, total])
            writer.writerow([model_name, difficulty, 'w/o reasoning', f"{accuracy_no_cot:.4f}", correct_no_cot, total])

    # Overall accuracy
    overall_accuracy_cot = total_correct_cot / total_questions if total_questions > 0 else 0.0
    overall_accuracy_no_cot = total_correct_no_cot / total_questions if total_questions > 0 else 0.0

    print("\n----------------------------------------")
    print("Overall Accuracy")
    print(f"  Accuracy (w/ reasoning): {overall_accuracy_cot:.4f} ({total_correct_cot}/{total_questions})")
    print(f"  Accuracy (w/o reasoning): {overall_accuracy_no_cot:.4f} ({total_correct_no_cot}/{total_questions})")
    print("----------------------------------------")

    # Write overall to CSV
    with open(csv_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([model_name, 'Overall', 'w/ reasoning', f"{overall_accuracy_cot:.4f}", total_correct_cot, total_questions])
        writer.writerow([model_name, 'Overall', 'w/o reasoning', f"{overall_accuracy_no_cot:.4f}", total_correct_no_cot, total_questions])


if __name__ == "__main__":
    results_path = os.path.join(os.path.dirname(__file__), '../results')
    model_names = [f for f in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, f))]
    
    if not model_names:
        print("No model result directories found in ../results")
    else:
        for model_name in model_names:
            compute_scores(model_name)
