import re
import numpy as np
import json
from typing import List

# Simplified parsing function
def parse_numbers(line: str) -> List[List[int]]:
    """
    Extract all numbers from the line and convert to nested list format.
    """
    line = line.strip()
    if not line:
        return []
    
    # Extract all numbers
    numbers = re.findall(r'\d+', line)
    if not numbers:
        return []
    
    # Convert to integer list
    int_list = [int(num) for num in numbers]
    
    # Check if original line contains multiple list structures
    if '] [' in line or '][[' in line:
        # Simple handling: return all numbers as a single list
        # This may not be completely correct, but handles most cases
        return [int_list]
    else:
        # Single list
        return [int_list]

# calculate_relaxed_match function copied from reference file
def calculate_relaxed_match(pred_lists: List[List[int]], gt_lists: List[List[int]]) -> float:
    """
    Calculate relaxed matching score as a product of precision scores for each sublist.

    For each predicted sublist and corresponding ground truth sublist:
    - Calculates precision as (number of predicted elements in ground truth) / (number of predicted elements)
    - Returns product of precision scores across all sublists

    Returns:
    - 0.0 if number of sublists don't match
    - Product of precision scores (between 0.0 and 1.0) otherwise
    """
    # Check if number of sublists match
    if len(pred_lists) != len(gt_lists):
        return 0.0

    # Check each corresponding sublist pair
    precision_all_goals = []
    for pred_sublist, gt_sublist in zip(pred_lists, gt_lists):
        # If none of the predicted elements appear in ground truth sublist, return 0
        if len(pred_sublist) == 0 and len(gt_sublist) == 0:
            precision = 1.0
        elif len(pred_sublist) == 0 or len(gt_sublist) == 0:
            precision = 0.0
        else:
            precision = sum(pred_elem in gt_sublist for pred_elem in pred_sublist) / len(pred_sublist)
            precision_all_goals.append(precision)

    # multiply precision of all goals
    return float(np.prod(precision_all_goals)) if precision_all_goals else 0.0

# Main function
def main():
    # Set file paths
    model_output_file = "/home/ccc/Documents/myCode/lifelong/myTest/frame/findingdory/output/val_answers_qwen1.5_7b_sft201_300_20epoch_2_merged_1114.jsonl"
    golden_answer_file = "/home/ccc/Documents/myCode/lifelong/myTest/frame/findingdory/output/val_golden_answer2.txt"
    
    # Read file contents
    try:
        # Read model output JSONL file, extract answer field
        model_outputs = []
        with open(model_output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        model_outputs.append(data.get('answer', ''))
                    except json.JSONDecodeError:
                        model_outputs.append('')
        
        # Read golden answer file
        with open(golden_answer_file, "r", encoding="utf-8") as f:
            golden_answers = [line.strip() for line in f]
            
        # Check if file line counts match
        if len(model_outputs) != len(golden_answers):
            print(f"Error: The number of lines in the two files does not match")
            print(f"Model output lines: {len(model_outputs)}")
            print(f"Golden answer lines: {len(golden_answers)}")
            return
            
        print(f"Read {len(model_outputs)} records")
        
        # Initialize counters
        relaxed_scores = []
        
        # Compare line by line
        for i, (model_output, golden_answer) in enumerate(zip(model_outputs, golden_answers)):
            # Remove newlines
            model_output = model_output.strip()
            golden_answer = golden_answer.strip()
            
            # Calculate relaxed accuracy
            pred_lists = parse_numbers(model_output)
            gt_lists = parse_numbers(golden_answer)
            
            relaxed_score = calculate_relaxed_match(pred_lists, gt_lists)
            relaxed_scores.append(relaxed_score)
            
        # Calculate final metrics
        avg_relaxed_accuracy = np.mean(relaxed_scores) if relaxed_scores else 0
        
        # Output results
        print(f"\n===== Evaluation Results =====")
        print(f"Average Relaxed Accuracy: {avg_relaxed_accuracy:.4f}")
        print(f"Total records: {len(model_outputs)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    
    