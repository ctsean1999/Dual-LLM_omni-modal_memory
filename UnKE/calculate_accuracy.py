import os
import json
import torch
import re
import numpy as np
import argparse
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set random seed to ensure reproducible results
def set_seed(seed=2024):
    np.random.seed(seed)

# Simple text preprocessing function
def preprocess_text(text):
    if not text:
        return ""
    # Remove special characters and extra spaces
    text = re.sub(r'[^一 - 龥 a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Define local model paths
local_models = {
    'all-MiniLM-L6-v2': 'Path to all-MiniLM-L6-v2',
    'bert-base-chinese': 'Path to bert-base-chinese',
    'chinese-roberta-wwm-ext': 'Path to chinese-roberta-wwm-ext'
}

# Load model and tokenizer
_model_cache = {}

def load_model_and_tokenizer(model_name):
    """Load specified model and tokenizer"""
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    try:
        if model_name == 'all-MiniLM-L6-v2':
            tokenizer = AutoTokenizer.from_pretrained(local_models[model_name])
            model = AutoModel.from_pretrained(local_models[model_name])
        else:
            tokenizer = BertTokenizer.from_pretrained(local_models[model_name])
            model = BertModel.from_pretrained(local_models[model_name])
        
        _model_cache[model_name] = (tokenizer, model)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return None, None

# Calculate text similarity using pre-trained model
def calculate_model_similarity(text1, text2, model_name='all-MiniLM-L6-v2'):
    """Calculate similarity between two texts using pre-trained model"""
    tokenizer, model = load_model_and_tokenizer(model_name)
    if tokenizer is None or model is None:
        return 0.0
    
    # Preprocess text
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    
    if not text1 or not text2:
        return 0.0
    
    device = torch.device('cuda')
    # device = torch.device('cpu')
    model.to(device)
    
    # Preprocess text and get tokens
    cand_tokens = tokenizer(text1, return_tensors='pt', padding=True, truncation=True, max_length=512)
    ref_tokens = tokenizer(text2, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Move to device
    cand_tokens = {k: v.to(device) for k, v in cand_tokens.items()}
    ref_tokens = {k: v.to(device) for k, v in ref_tokens.items()}
    
    # Get hidden states
    with torch.no_grad():
        cand_output = model(**cand_tokens)
        ref_output = model(**ref_tokens)
    
    # Get hidden states from last layer
    cand_embeddings = cand_output.last_hidden_state[0]
    ref_embeddings = ref_output.last_hidden_state[0]
    
    # Calculate cosine similarity matrix
    cos_sim = torch.nn.functional.cosine_similarity(
        cand_embeddings.unsqueeze(1), 
        ref_embeddings.unsqueeze(0), 
        dim=2
    )
    
    # Calculate precision, recall and F1
    precision = torch.max(cos_sim, dim=1)[0].mean().item()
    recall = torch.max(cos_sim, dim=0)[0].mean().item()
    
    # Use F1 score as final similarity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def calculate_accuracy(results_file, dataset_file, similarity_model='all-MiniLM-L6-v2'):
    """Calculate accuracy between model answers and original answers"""
    # Set random seed
    set_seed()
    
    # File paths
    print(f"Using results file: {results_file}")
    print(f"Using dataset file: {dataset_file}")
    
    # Check if files exist
    if not os.path.exists(results_file):
        print(f"Error: Model results file not found - {results_file}")
        return
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found - {dataset_file}")
        return
    
    # Load dataset
    print("Loading dataset file...")
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Successfully loaded dataset, containing {len(dataset)} videos")
    except Exception as e:
        print(f"Error loading dataset file: {str(e)}")
        return
    
    # Create id to answer mappings
    id_to_answer = {}
    id_to_sub_answers = {}  # Mapping for sub_question answers
    id_to_mmlu = {}  # Mapping for mmlu questions and answers
    for item in dataset:
        if 'id' in item and 'answer' in item:
            id_to_answer[item['id']] = item['answer']
        
        # Create mapping from sub_question to sub_answer
        if 'id' in item and 'sub_question' in item and 'sub_answer' in item:
            sub_questions = item['sub_question']
            sub_answers = item['sub_answer']
            
            # Create list of question-answer pairs
            qa_pairs = []
            for i, question in enumerate(sub_questions):
                if i < len(sub_answers):
                    qa_pairs.append({
                        'question': question,
                        'answer': sub_answers[i]
                    })
            id_to_sub_answers[item['id']] = qa_pairs
        
        # Create mapping for mmlu questions and answers
        if 'id' in item and 'mmlu_questions' in item and 'mmlu_answer' in item:
            mmlu_questions = item['mmlu_questions']
            mmlu_answers = item['mmlu_answer']
            
            # Create mapping from question text to correct answer
            question_to_answer = {}
            for i, question_text in enumerate(mmlu_questions):
                if i < len(mmlu_answers):
                    question_to_answer[question_text] = mmlu_answers[i]
            id_to_mmlu[item['id']] = question_to_answer
    
    # Load model results
    print("Loading model results file...")
    model_results = []
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'results' in data:
                model_results = data['results']
            else:
                print("Error: JSON file does not contain 'results' key")
                return
        print(f"Successfully loaded model results, containing {len(model_results)} results")
    except Exception as e:
        print(f"Error loading model results file: {str(e)}")
        return
    
    # Calculate accuracy
    print("Starting accuracy calculation...")
    similarities = []  # Store similarity between model_answer and original_answer
    paraphrased_similarities = []  # Store similarity between paraphrased_answer and original_answer
    sub_question_similarities = []  # Store similarity between model_answer and expected_answer for each sub_question
    mmlu_correct_count = 0  # Store number of correct mmlu_answers
    mmlu_total_count = 0  # Store total number of mmlu_answers
    matched_count = 0
    paraphrased_matched_count = 0
    sub_question_matched_count = 0
    
    for i, result in enumerate(model_results):
        video_id = result.get('id')
        model_answer = result.get('answer')
        paraphrased_answer = result.get('para_answer')
        sub_questions = result.get('sub_questions')
        
        if video_id is None:
            print(f"Warning: Result missing id field")
            continue
        
        if video_id not in id_to_answer:
            print(f"Warning: Video ID {video_id} not in dataset")
            continue
        
        original_answer = id_to_answer[video_id]
        
        # Calculate similarity between model_answer and original_answer
        if model_answer is not None:
            similarity = calculate_model_similarity(original_answer, model_answer, similarity_model)
            similarities.append(similarity)
            matched_count += 1
        
        # Calculate similarity between paraphrased_answer and original_answer
        if paraphrased_answer is not None:
            paraphrased_similarity = calculate_model_similarity(original_answer, paraphrased_answer, similarity_model)
            paraphrased_similarities.append(paraphrased_similarity)
            paraphrased_matched_count += 1
        
        # Calculate similarity between model_answer and expected_answer for each sub_question
        if sub_questions is not None and isinstance(sub_questions, list):
            for j, question in enumerate(sub_questions):
                question_model_answer = question.get('answer')
                # Get answer field from sub_answers in dataset_file
                question_expected_answer = None
                if video_id in id_to_sub_answers and j < len(id_to_sub_answers[video_id]):
                    question_expected_answer = id_to_sub_answers[video_id][j].get('answer')

                if question_model_answer is not None and question_expected_answer is not None:
                    sub_similarity = calculate_model_similarity(question_expected_answer, question_model_answer, similarity_model)
                    sub_question_similarities.append(sub_similarity)
                    sub_question_matched_count += 1
        
        # Calculate if model_answer matches expected_answer for each question in mmlu_results
        mmlu_results = result.get('mmlu_results')
        if mmlu_results is not None and isinstance(mmlu_results, list) and video_id in id_to_mmlu:
            question_to_answer = id_to_mmlu[video_id]
            for mmlu_result in mmlu_results:
                question_text = mmlu_result.get('question')
                question_model_answer = mmlu_result.get('model_answer')
                
                if question_text and question_model_answer is not None and question_text in question_to_answer:
                    expected_answer = question_to_answer[question_text]
                    mmlu_total_count += 1
                    # Compare if values are the same
                    if str(question_model_answer).strip() == str(expected_answer).strip():
                        mmlu_correct_count += 1
        
        # Show progress every 10 results processed
        if (i + 1) % 10 == 0:
            print(f"{i + 1} results processed")
    
    # Statistics for similarity between model_answer and original_answer
    print(f"\nSimilarity calculation between model_answer and original_answer completed!")
    if similarities:
        average_similarity = sum(similarities) / len(similarities)
        print(f"Number of successfully matched results: {matched_count}")
        print(f"Average similarity of all results: {average_similarity:.4f}")
        print(f"Maximum similarity: {max(similarities):.4f}")
        print(f"Minimum similarity: {min(similarities):.4f}")
    else:
        print(f"No matching results found, cannot calculate similarity")
    
    # Statistics for similarity between paraphrased_answer and original_answer
    print(f"\nSimilarity calculation between paraphrased_answer and original_answer completed!")
    if paraphrased_similarities:
        average_paraphrased_similarity = sum(paraphrased_similarities) / len(paraphrased_similarities)
        print(f"Number of successfully matched results: {paraphrased_matched_count}")
        print(f"Average similarity of all results: {average_paraphrased_similarity:.4f}")
        print(f"Maximum similarity: {max(paraphrased_similarities):.4f}")
        print(f"Minimum similarity: {min(paraphrased_similarities):.4f}")
    else:
        print(f"No matching results found, cannot calculate similarity")
    
    # Statistics for similarity between model_answer and expected_answer for sub_questions
    print(f"\nSimilarity calculation between model_answer and expected_answer for sub_questions completed!")
    if sub_question_similarities:
        average_sub_similarity = sum(sub_question_similarities) / len(sub_question_similarities)
        print(f"Number of successfully matched questions: {sub_question_matched_count}")
        print(f"Average similarity of all results: {average_sub_similarity:.4f}")
        print(f"Maximum similarity: {max(sub_question_similarities):.4f}")
        print(f"Minimum similarity: {min(sub_question_similarities):.4f}")
    else:
        print(f"No matching questions found, cannot calculate similarity")
    
    # Statistics for accuracy between model_answer and expected_answer for mmlu_results
    print(f"\nAccuracy calculation between model_answer and expected_answer for mmlu_results completed!")
    if mmlu_total_count > 0:
        accuracy = mmlu_correct_count / mmlu_total_count
        print(f"Number of successfully matched questions: {mmlu_total_count}")
        print(f"Number of correct questions: {mmlu_correct_count}")
        print(f"Average accuracy: {accuracy:.4f}")
    else:
        print(f"No matching questions found, cannot calculate accuracy")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate accuracy between model answers and original answers')
    
    # Results file path
    parser.add_argument('--results_file', 
                      default='./model_results.json',
                      help='Path to the model results JSON file')
    
    # Dataset file path
    parser.add_argument('--dataset_file', 
                      default='Path to UnKE dataset ./final_data_v3.json',
                      help='Path to the dataset annotation JSON file')
    
    # Similarity model name
    parser.add_argument('--similarity_model', 
                      default='all-MiniLM-L6-v2',
                      choices=['all-MiniLM-L6-v2', 'bert-base-chinese', 'chinese-roberta-wwm-ext'],
                      help='Model to use for calculating similarity')
    
    args = parser.parse_args()
    
    print("Start calculating accuracy between model answers and original answers...")
    print(f"Using model: {args.similarity_model} ({local_models[args.similarity_model]})")
    print("Read model answers, paraphrased answers, sub_questions, and MMLU results from JSON file; read original answers and MMLU questions from JSON file")
    print("Calculate answer similarity for each sample and compute average")
    print("=" * 60)
    
    calculate_accuracy(args.results_file, args.dataset_file, args.similarity_model)

if __name__ == '__main__':
    main()
