import ast
import json
import numpy as np
from typing import List, Dict
from itertools import permutations

# Functions copied from reference file

def parse_list_string(list_string: str) -> List[List[int]]:
    """
    Convert string representation of list of lists to actual nested list.
    Handles various formats:
    - Space-separated numbers (e.g., "[62 63 64]" instead of "[62, 63, 64]")
    - Lines without brackets (e.g., "17 18 19 20...")
    - Nested lists with brackets
    - Truncated lines
    """
    import re
    list_string = list_string.strip()
    
     # print(f"Original input string: '{list_string}'")
    
    # Handle empty strings
    if not list_string:
        # print("Input string is empty, returning empty list")
        return []
    
    try:
        # Replace all space-separated numbers with comma-separated
        list_string = re.sub(r'(\d+)\s+(\d+)', r'\1, \2', list_string)
        # print(f"String after replacing spaces: '{list_string}'")
        
        # Check if string has brackets
        if not (list_string.startswith('[') and list_string.endswith(']')):
            # If no brackets, try to wrap in brackets
            # First check if it contains multiple lists (has a closing bracket in the middle)
            if ']' in list_string:
                # Add outer brackets
                list_string = f"[{list_string}]"
                # print(f"Added outer brackets: '{list_string}'")
            else:
                # Single list without brackets, wrap in double brackets
                list_string = f"[[{list_string}]]"
                # print(f"Added double brackets: '{list_string}'")
        
        # Check if it's a single list (needs to be wrapped in another list)
        parsed = ast.literal_eval(list_string)
        # print(f"Parsed result: {parsed}")
        
        if isinstance(parsed, list):
            if not parsed:
                # print("Parsed result is empty list, returning empty list")
                return []
            if isinstance(parsed[0], int):
                # Single list, wrap in another list
                result = [parsed]
                # print(f"Single list, wrapped: {result}")
                return result
            elif isinstance(parsed[0], list):
                # Already nested list
                # print(f"Already nested list: {parsed}")
                return parsed
            else:
                # print(f"Unrecognized format: {parsed}")
                return []
        return []
    except Exception as e:
        # print(f"Parsing failed: {e}")
        try:
            # If parsing fails, try to extract all numbers and create a list
            numbers = re.findall(r'\d+', list_string)
            # print(f"Extracted numbers: {numbers}")
            
            if not numbers:
                # print("No numbers found, returning empty list")
                return []
            
            # Convert to integers
            int_list = [int(num) for num in numbers]
            # print(f"Converted integer list: {int_list}")
            
            # Try to determine structure based on original string
            if '] [' in list_string or '][[' in list_string:
                # Probably multiple lists, but we can't be sure, so return as single list
                # This is a fallback
                result = [int_list]
                # print(f"Probably multiple lists, returning single wrapped list: {result}")
                return result
            else:
                # Single list
                result = [int_list]
                # print(f"Single list, returning: {result}")
                return result
        except Exception as e2:
            # print(f"Final parsing failed: {e2}")
            # Final fallback: return empty list
            return []

def calculate_max_relaxed_match(pred, gold):
    """
    Calculate the highest relaxed match average between pred and gold arrays.
    
    Parameters:
    pred -- Prediction array, containing multiple subarrays, each containing integer elements
    gold -- Ground truth array, containing multiple subarrays, each containing integer elements
    
    Returns:
    Highest relaxed match average
    """
    # print(f"\nCalculating Max Relaxed Match")
    # print(f"Prediction array: {pred}")
    # print(f"Ground truth array: {gold}")
    
    # Check if input is valid
    if not pred or not gold:
        # print("Prediction array or ground truth array is empty, returning 0.0")
        return 0.0
    
    # Function to calculate relaxed match
    def relaxed_match(arr_a, arr_b):
        """Calculate relaxed match average for array A against array B"""
        # print(f"  Calculating Relaxed Match: {arr_a} vs {arr_b}")
        
        if not arr_a:
            # print("  Array A is empty, returning 0.0")
            return 0.0
        
        # Convert arr_b to set for faster lookup
        set_b = set(arr_b)
        # print(f"  Array B converted to set: {set_b}")
        
        # Count elements in arr_a that appear in arr_b
        count = sum(1 for elem in arr_a if elem in set_b)
        # print(f"  Matching elements count: {count}/{len(arr_a)}")
        
        # Return ratio
        result = count / len(arr_a) if len(arr_a) > 0 else 0.0
        # print(f"  Relaxed Match result: {result}")
        return result
    
    # Get lengths of pred and gold
    n_pred = len(pred)
    n_gold = len(gold)
    # print(f"Prediction array length: {n_pred}, Ground truth array length: {n_gold}")
    
    # If pred or gold is empty, return 0
    if n_pred == 0 or n_gold == 0:
        # print("Prediction array or ground truth array is empty, returning 0.0")
        return 0.0
    
    # Create cost matrix (using 1-relaxed_match as cost, since we want to maximize match values)
    # Ensure matrix is square, add dummy rows/columns if needed
    max_size = max(n_pred, n_gold)
    cost_matrix = np.zeros((max_size, max_size))
    
    # Fill actual values
    for i in range(n_pred):
        for j in range(n_gold):
            match_value = relaxed_match(pred[i], gold[j])
            cost_matrix[i, j] = 1.0 - match_value  # Convert to cost
    
    # Fill dummy rows/columns (if pred and gold have different lengths)
    # Dummy rows have cost 1 (match value 0)
    for i in range(n_pred, max_size):
        cost_matrix[i, :] = 1.0
    
    for j in range(n_gold, max_size):
        cost_matrix[:, j] = 1.0
    
    # Use Hungarian algorithm to find optimal assignment
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Calculate actual match average
        total_match = 0.0
        count = 0
        
        for i, j in zip(row_ind, col_ind):
            if i < n_pred and j < n_gold:  # Ignore dummy rows/columns
                match_value = 1.0 - cost_matrix[i, j]  # Convert back to match value
                total_match += match_value
                count += 1
        
        avg_match = total_match / count if count > 0 else 0.0
        # print(f"Final Max Relaxed Match result: {avg_match}")
        return avg_match
        
    except ImportError:
        # If scipy is not available, use simplified greedy algorithm
        # print("Warning: scipy not installed, using simplified algorithm")
        
        # Ensure pred is the shorter array
        if n_pred > n_gold:
            pred, gold = gold, pred
            n_pred, n_gold = n_gold, n_pred
        
        # Greedy matching: find best match pair each time
        used_gold = set()
        total_match = 0.0
        
        for i in range(n_pred):
            best_match = 0.0
            best_j = -1
            
            for j in range(n_gold):
                if j not in used_gold:
                    match_value = relaxed_match(pred[i], gold[j])
                    if match_value > best_match:
                        best_match = match_value
                        best_j = j
            
            if best_j != -1:
                used_gold.add(best_j)
                total_match += best_match
        
        avg_match = total_match / n_pred if n_pred > 0 else 0.0
        # print(f"Final Max Relaxed Match result: {avg_match}")
        return avg_match


def calculate_max_exact_match(pred, gold):
    """
    Calculate the highest exact match value between pred and gold arrays.
    
    Parameters:
    pred -- Prediction array, containing multiple subarrays, each containing integer elements
    gold -- Ground truth array, containing multiple subarrays, each containing integer elements
    
    Returns:
    Highest exact match value
    """
    # print(f"\nCalculating Max Exact Match")
    # print(f"Prediction array: {pred}")
    # print(f"Ground truth array: {gold}")
    
    # Check if input is valid
    if not pred or not gold:
        # print("Prediction array or ground truth array is empty, returning 0.0")
        return 0.0
    
    # Function to calculate exact match
    def exact_match(arr_a, arr_b):
        """Calculate exact match value for array A against array B"""
        # print(f"  Calculating Exact Match: {arr_a} vs {arr_b}")
        
        if not arr_a or not arr_b:
            # print("  Array A or B is empty, returning 0.0")
            return 0.0
        
        # Convert arrays to sets for comparison
        set_a = set(arr_a)
        set_b = set(arr_b)
        # print(f"  Array A set: {set_a}")
        # print(f"  Array B set: {set_b}")
        
        # If both sets are identical, return 1, otherwise return 0
        result = 1.0 if set_a == set_b else 0.0
        # print(f"  Exact Match result: {result}")
        return result
    
    # Get lengths of pred and gold
    n_pred = len(pred)
    n_gold = len(gold)
    # print(f"Prediction array length: {n_pred}, Ground truth array length: {n_gold}")
    
    # If pred or gold is empty, return 0
    if n_pred == 0 or n_gold == 0:
        # print("Prediction array or ground truth array is empty, returning 0.0")
        return 0.0
    
    # Create cost matrix (using 1-exact_match as cost, since we want to maximize match values)
    # Ensure matrix is square, add dummy rows/columns if needed
    max_size = max(n_pred, n_gold)
    cost_matrix = np.zeros((max_size, max_size))
    
    # Fill actual values
    for i in range(n_pred):
        for j in range(n_gold):
            match_value = exact_match(pred[i], gold[j])
            cost_matrix[i, j] = 1.0 - match_value  # Convert to cost
    
    # Fill dummy rows/columns (if pred and gold have different lengths)
    # Dummy rows have cost 1 (match value 0)
    for i in range(n_pred, max_size):
        cost_matrix[i, :] = 1.0
    
    for j in range(n_gold, max_size):
        cost_matrix[:, j] = 1.0
    
    # Use Hungarian algorithm to find optimal assignment
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Calculate actual match average
        total_match = 0.0
        count = 0
        
        for i, j in zip(row_ind, col_ind):
            if i < n_pred and j < n_gold:  # Ignore dummy rows/columns
                match_value = 1.0 - cost_matrix[i, j]  # Convert back to match value
                total_match += match_value
                count += 1
        
        avg_match = total_match / count if count > 0 else 0.0
        # print(f"Final Max Exact Match result: {avg_match}")
        return avg_match
        
    except ImportError:
        # If scipy is not available, use simplified greedy algorithm
        # print("Warning: scipy not installed, using simplified algorithm")
        
        # Ensure pred is the shorter array
        if n_pred > n_gold:
            pred, gold = gold, pred
            n_pred, n_gold = n_gold, n_pred
        
        # Greedy matching: find best match pair each time
        used_gold = set()
        total_match = 0.0
        
        for i in range(n_pred):
            best_match = 0.0
            best_j = -1
            
            for j in range(n_gold):
                if j not in used_gold:
                    match_value = exact_match(pred[i], gold[j])
                    if match_value > best_match:
                        best_match = match_value
                        best_j = j
            
            if best_j != -1:
                used_gold.add(best_j)
                total_match += best_match
        
        avg_match = total_match / n_pred if n_pred > 0 else 0.0
        # print(f"Final Max Exact Match result: {avg_match}")
        return avg_match


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
                        
        with open(golden_answer_file, "r", encoding="utf-8") as f:
            golden_answers = f.readlines()
            
        # print(f"Model output lines: {len(model_outputs)}")
        # print(f"Golden answer lines: {len(golden_answers)}")
        # print(f"Processing {len(model_outputs)} records")
        
        # Initialize counters
        max_relaxed_scores = []
        max_exact_scores = []
        
        # Compare line by line, only process lines in model_outputs
        for i in range(len(model_outputs)):
            # print(f"\n===== Processing record {i+1} =====")
            model_output = model_outputs[i].strip()
            # print(f"Model output: '{model_output}'")
            
            # If golden_answers has corresponding line, use it, otherwise use empty string
            golden_answer = golden_answers[i].strip()
            # print(f"Golden answer: '{golden_answer}'")
            
            # Calculate relaxed accuracy
            pred_lists = parse_list_string(model_output)
            gt_lists = parse_list_string(golden_answer)
            
            # print(f"Parsed prediction lists: {pred_lists}")
            # print(f"Parsed golden answer lists: {gt_lists}")
            
            # Calculate highest relaxed match accuracy
            max_relaxed_score = calculate_max_relaxed_match(pred_lists, gt_lists)
            max_relaxed_scores.append(max_relaxed_score)
            print(f"Record {i+1} Max Relaxed Score: {max_relaxed_score}")
            
        # Calculate final metrics
        avg_max_relaxed_accuracy = np.mean(max_relaxed_scores) 
        
        # Output results
        print(f"\n===== Evaluation Results =====")
        print(f"Average Max Relaxed Accuracy: {avg_max_relaxed_accuracy:.4f}")
        print(f"Total records: {len(model_outputs)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()