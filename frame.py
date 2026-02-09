import os
import json
import torch
import numpy as np
import csv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BertTokenizer, BertModel
import time
from datetime import datetime
from openai import OpenAI

# Set random seed for reproducible results
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set fixed seed
set_seed(42)

# Load dataset function
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Load Qwen1.5 model
def load_qwen_model(model_path):
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float32,
        trust_remote_code=True
    )
    print(f"Model {model_path} loaded successfully")
    return tokenizer, model

# Generate answer function
def generate_answer(tokenizer, model, question, max_tokens=1024):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Ensure using fixed random seed
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.01,  # 降低温度参数，减少随机性
            do_sample=False,   # 不使用采样，使用贪婪解码
            top_k=1,           # 仅考虑概率最高的1个token
            top_p=0.01,        # 设置极低的top_p值
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Similarity calculation functions
# Local model paths
local_models = {
    'chinese-roberta-wwm-ext': '/home/ccc/Documents/myCode/LLM/chinese-roberta-wwm-ext'
}

# Simple text preprocessing function
def preprocess_text(text):
    if not text:
        return ""
    import re
    # Remove special characters and extra spaces
    text = re.sub(r'[^一-龥a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Model cache
_model_cache = {}

# Load model and tokenizer
def load_model_and_tokenizer(model_name):
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    try:
        tokenizer = BertTokenizer.from_pretrained(local_models[model_name])
        model = BertModel.from_pretrained(local_models[model_name])
        
        _model_cache[model_name] = (tokenizer, model)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return None, None

# Calculate text similarity using pre-trained model
def calculate_model_similarity(text1, text2, model_name='chinese-roberta-wwm-ext'):
    tokenizer, model = load_model_and_tokenizer(model_name)
    if tokenizer is None or model is None:
        return 0.0
    
    # Preprocess text
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    
    if not text1 or not text2:
        return 0.0
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # Get last layer hidden states
    cand_embeddings = cand_output.last_hidden_state[0]
    ref_embeddings = ref_output.last_hidden_state[0]
    
    # Calculate cosine similarity matrix
    cos_sim = torch.nn.functional.cosine_similarity(
        cand_embeddings.unsqueeze(1), 
        ref_embeddings.unsqueeze(0), 
        dim=2
    )
    
    # Calculate precision, recall, and F1
    precision = torch.max(cos_sim, dim=1)[0].mean().item()
    recall = torch.max(cos_sim, dim=0)[0].mean().item()
    
    # Use F1 score as final similarity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

# Configure OpenAI compatible client
def get_openai_client():
    return OpenAI(
        # Read API Key from environment variable
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

# Call Qwen3-Max model to get answer with RAG support
def get_model_answer_with_rag(question, reference, client):
    try:
        # Construct prompt with reference materials
        if reference:
            prompt = f"User's question: {question}\n\nPlease answer by referring to the following materials: {reference}\n\nAnswer as briefly as possible, no more than 100 words."
        else:
            prompt = f"{question}\n\nAnswer as briefly as possible, no more than 100 words."
        
        # Construct OpenAI format messages
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Call OpenAI compatible API
        response = client.chat.completions.create(
            model="qwen3-max",  # Qwen3-Max模型
            messages=messages,
            temperature=0.6,
        )
        
        # Extract answer content
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
        else:
            return "Failed to get answer"
            
    except Exception as e:
        print(f"Error calling model: {str(e)}")
        return f"Error: {str(e)}"

# Main function
def main():
    # Dataset path
    # data_path = '/home/ccc/Documents/myCode/lifelong/myTest/frame/data/common_question.json'
    data_path = '/home/ccc/Documents/myCode/lifelong/myTest/frame/data/NeijingClipsMultimodalDataset339.json'
    
    # Get filename from data_path
    data_filename = os.path.basename(data_path)
    
    # Model paths
    model_paths = [
        '/home/ccc/Documents/myCode/LLM/Qwen1.5-7B-Chat',
        '/home/ccc/Documents/myCode/lifelong/myTest/frame/model/qwen1_5_7b_pretrain96epoch_merged_lora12epoch_merged'
    ]
    
    # Load dataset
    print(f"Loading dataset: {data_path}")
    data = load_dataset(data_path)[:2]
    print(f"Successfully loaded {len(data)} records")
    
    # Load two Qwen models
    tokenizers = []
    models = []
    for model_path in model_paths:
        tokenizer, model = load_qwen_model(model_path)
        tokenizers.append(tokenizer)
        models.append(model)
    
    # Initialize OpenAI client
    client = get_openai_client()
    
    # Initialize total scores
    total_scores = []
    
    # Prepare CSV file to save results
    csv_file = '/home/ccc/Documents/myCode/lifelong/myTest/frame/result/NeijingClipsMultimodalDataset339_two_model.csv'
    
    # Check if file exists, create and write header if not
    file_exists = os.path.isfile(csv_file)
    if not file_exists:
        # Create CSV file and write header
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Remove question ID column, add original answer column after question content, adjust position of LLM answer column
            writer.writerow(['Question Content', 'Original Answer', 'Model 1 Answer', 'Model 2 Answer', 'chinese-roberta-wwm-ext Similarity', 'LLM Answer', 'Timestamp', 'Data Filename'])
        print(f"Result file created: {csv_file}")
    else:
        print(f"Appending data to existing file: {csv_file}")
    
    print(f"Results will be saved to: {csv_file}")
    
    # Process each question
    for i, item in enumerate(data):
        print(f"Processing record {i+1}/{len(data)}")
        
        question = item.get('question', '')
        question = question + " \n  Please answer as briefly as possible, no more than 100 words."
        if not question:
            print(f"Skipping record {i+1} because question is empty")
            continue
        
        # Generate answers for each model
        model_answers = []
        for j, (tokenizer, model) in enumerate(zip(tokenizers, models)):
            print(f"  Generating answer using model {j+1}...")
            try:
                answer = generate_answer(tokenizer, model, question)
                model_answers.append(answer)
                print(f"  Model {j+1} answer generated successfully")
            except Exception as e:
                print(f"  Error generating answer with model {j+1}: {str(e)}")
                model_answers.append("")
        
        # Calculate similarity between the two model answers
        similarity = 0.0
        if len(model_answers) >= 2 and model_answers[0] and model_answers[1]:
            print(f"  Calculating similarity between the two model answers...")
            try:
                # Use calculate_model_similarity to compute similarity between the two model answers
                similarity = calculate_model_similarity(model_answers[0], model_answers[1], 'chinese-roberta-wwm-ext')
                print(f"    chinese-roberta-wwm-ext similarity: {similarity:.4f}")
                total_scores.append(similarity)
            except Exception as e:
                print(f"    Error calculating chinese-roberta-wwm-ext similarity: {str(e)}")
                similarity = 0.0
        else:
            print(f"  Skipping similarity calculation because at least one model answer is empty")

        # Decide whether to add reference materials
        reference = ""
        if similarity < 0.7 and len(model_answers) >= 2:
            # Use the answer from the specified model (model/qwen1_5_7b_pretrain96epoch_merged_lora12epoch_merged) as reference
            reference = model_answers[1]  # Second model is the specified model
            print(f"  Similarity less than 0.7, adding reference material")
        else:
            print(f"  Similarity greater than or equal to 0.7 or insufficient model answers, not adding reference material")

        # Call LLM to answer the question
        print(f"  Calling LLM to answer the question...")
        try:
            llm_answer = get_model_answer_with_rag(question, reference, client)
            print(f"  LLM answer generated successfully")
        except Exception as e:
            print(f"  Error calling LLM: {str(e)}")
            llm_answer = ""

        # Get current time as timestamp
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get original answer
        original_answer = item.get('answer', '')
        
        # Write results to CSV file (append mode), regardless of whether similarity was calculated successfully
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                question,  # Question content
                original_answer,  # Original answer
                model_answers[0] if len(model_answers) > 0 else "",  # Model 1 answer
                model_answers[1] if len(model_answers) > 1 else "",  # Model 2 answer
                similarity,  # chinese-roberta-wwm-ext similarity
                llm_answer,  # LLM answer (moved after similarity)
                current_time,  # Timestamp
                data_filename  # Data filename
            ])
        print(f"  Results saved for record {i+1}")
        
        # Add delay to avoid issues from consecutive requests
        time.sleep(2)
    
    # Calculate overall average score
    if total_scores:
        overall_avg = sum(total_scores) / len(total_scores)
        print(f"\nOverall average similarity score of all model answers: {overall_avg:.4f}")
    else:
        print("\nNo similarity scores were successfully calculated")

if __name__ == '__main__':
    main()