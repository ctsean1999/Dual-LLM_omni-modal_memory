import os
import sys
import json
import torch
from tqdm import tqdm
import time
import random
import logging
import argparse

# Adjust Python module search path to ensure using newer packages from virtual environment
# Get site-packages directory of virtual environment
venv_site_packages = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'lib', 'python' + sys.version[:3], 'site-packages')
if venv_site_packages in sys.path:
    # If virtual environment site-packages is already in path, move it to the front
    sys.path.remove(venv_site_packages)
    sys.path.insert(0, venv_site_packages)
else:
    # If not in path, add to the front
    sys.path.insert(0, venv_site_packages)

# Import model and tokenizer from modelscope
from modelscope import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_analysis.log')
    ]
)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize model and tokenizer
def initialize_model(model_name):
    """
    Initialize model and tokenizer
    """
    logger.info("Loading model and tokenizer...")
    logger.info(f"Using model path: {model_name}")
    
    try:
        # Load model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            dtype="auto", 
            device_map="auto"
        )
        logger.info(f"✓ Model loaded successfully, type: {type(model).__name__}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"✓ Tokenizer loaded successfully, type: {type(tokenizer).__name__}")
        logger.info(f"Tokenizer's pad_token: {tokenizer.pad_token}")
        logger.info(f"Tokenizer's eos_token: {tokenizer.eos_token}")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model and tokenizer: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_metadata(json_file):
    """
    Load metadata
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"Successfully loaded metadata, total {len(metadata)} entries")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return []

def answer_question(question, choices=None, model=None, tokenizer=None, max_retries=3):
    """
    Answer question based on question text
    
    Args:
        question: Question text
        choices: Optional, list of multiple choice options
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer
        max_retries: Maximum number of retries for generation
    
    Returns:
        Text answered by the large model
    """
    try:
        # Build prompt
        prompt = f"Please answer the following question: {question}\n"
        
        # If there are choices, add them and explicitly require only outputting numbers 0-3
        if choices and isinstance(choices, list):
            prompt += "Options:\n"
            for i, choice in enumerate(choices):
                prompt += f"{i}. {choice}\n"
            prompt += "\nPlease strictly output only one of the numbers 0, 1, 2, 3 as the answer, do not output any other content."
        
        # Retry mechanism
        for attempt in range(max_retries):
            try:
                # Build messages
                messages = [
                    {
                        "role": "system",
                        "content": "You are a traditional Chinese medicine expert, proficient in Huangdi Neijing and traditional Chinese medicine, please provide the corresponding answer according to the question."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ]
                
                # Apply chat template
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                # Generate response
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512,  # Adjust to appropriate length
                    use_cache=True,  # Use cache to speed up generation
                    temperature=0.0,
                    do_sample=False,
                    # top_p=1.0,
                    # top_k=0,
                    # repetition_penalty=1.0
                )
                # Extract response text
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                return response_text
                
            except Exception as e:
                logger.error(f"Failed to answer question request (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff strategy
                    backoff_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(backoff_time)
                else:
                    return f"Failed to answer question: {str(e)}"
        
    except Exception as e:
        logger.error(f"Error occurred when answering question: {e}")
        import traceback
        traceback.print_exc()
        return f"Error occurred when answering question: {str(e)}"

def save_result_to_jsonl(result, jsonl_file):
    """
    Save analysis result to jsonl file
    """
    with open(jsonl_file, 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

def main():
    """
    Main function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze content using Qwen1.5-7B model')
    
    # Model path
    parser.add_argument('--model_path', 
                      default='./model/Qwen1.5-7B_findingdory_neijing_unke_final',
                      help='Path to the pre-trained model')
    
    # Output file path
    parser.add_argument('--output_file', 
                      default='./model_result.jsonl',
                      help='Path to save the analysis results in JSONL format')
    
    # Data file path
    parser.add_argument('--data_file', 
                      default='./annotation.json',
                      help='Path to the annotation data JSON file')
    
    # Maximum number of retries
    parser.add_argument('--max_retries', 
                      type=int, 
                      default=3,
                      help='Maximum number of retries for API calls')
    
    args = parser.parse_args()
    
    try:
        # Check model path
        if not os.path.exists(args.model_path):
            print(f"Error: Model path does not exist: {args.model_path}")
            logger.error(f"Model path does not exist: {args.model_path}")
            return
        
        # Initialize model and tokenizer
        model, tokenizer = initialize_model(args.model_path)
        
        # Load metadata
        metadata = load_metadata(args.data_file)
        
        # Read processed content IDs
        processed_ids = set()
        if os.path.exists(args.output_file):
            try:
                with open(args.output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                result = json.loads(line)
                                if 'content_id' in result:
                                    processed_ids.add(result['content_id'])
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"Failed to read output file: {e}")
        
        logger.info(f"Number of processed contents: {len(processed_ids)}")
        
        # Analyze content entries one by one
        for content_info in tqdm(metadata, desc="Processing content"):
            content_id = content_info.get('id', '')
            
            # Check if content has already been processed
            if content_id in processed_ids:
                continue
            
            logger.info(f"Processing content: {content_id}")
            
            # Answer all question types
            results = {
                "content_id": content_id
            }
            
            # 1. Answer original question
            if 'question' in content_info and content_info['question']:
                results["question"] = content_info['question']
                results["answer"] = answer_question(content_info['question'], model=model, tokenizer=tokenizer, max_retries=args.max_retries)
            
            # 2. Answer paraphrased question
            if 'paraphrased_question' in content_info and content_info['paraphrased_question']:
                results["paraphrased_question"] = content_info['paraphrased_question']
                results["paraphrased_answer"] = answer_question(content_info['paraphrased_question'], model=model, tokenizer=tokenizer, max_retries=args.max_retries)
            
            # 3. Answer all questions in multi-hop QA
            if 'multihop_qa' in content_info and isinstance(content_info['multihop_qa'], list):
                results["multihop_qa"] = []
                for hop_item in content_info['multihop_qa']:
                    if 'question' in hop_item and hop_item['question']:
                        hop_answer = answer_question(hop_item['question'], model=model, tokenizer=tokenizer, max_retries=args.max_retries)
                        results["multihop_qa"].append({
                            "question": hop_item['question'],
                            "model_answer": hop_answer
                        })
            
            # 4. Answer MMLU multiple choice questions
            if ('mmlu_questions' in content_info and isinstance(content_info['mmlu_questions'], list) and
                'mmlu_choices' in content_info and isinstance(content_info['mmlu_choices'], list)):
                # Ensure number of questions and choices match
                if len(content_info['mmlu_questions']) == len(content_info['mmlu_choices']):
                    results["mmlu_results"] = []
                    for i in range(len(content_info['mmlu_questions'])):
                        mmlu_question = content_info['mmlu_questions'][i]
                        mmlu_question_choices = content_info['mmlu_choices'][i]
                        
                        if mmlu_question:
                            # Ensure choices format is correct
                            if not (isinstance(mmlu_question_choices, list) and len(mmlu_question_choices) == 4):
                                if not isinstance(mmlu_question_choices, list):
                                    mmlu_question_choices = []
                                # Pad or truncate choices
                                mmlu_question_choices = mmlu_question_choices[:4]
                                while len(mmlu_question_choices) < 4:
                                    mmlu_question_choices.append("")
                            
                            # Answer validation and retry logic for MMLU multiple choice questions
                            valid_answer = False
                            attempt = 0
                            model_answer = None
                            
                            while not valid_answer and attempt < args.max_retries:
                                attempt += 1
                                model_answer = answer_question(mmlu_question, choices=mmlu_question_choices, model=model, tokenizer=tokenizer, max_retries=1)  # Only 1 retry here since we have our own retry loop
                                
                                # Validate if answer is one of 0, 1, 2, 3
                                if model_answer.strip() in ['0', '1', '2', '3']:
                                    valid_answer = True
                                    logger.info(f"MMLU multiple choice answer is valid: {model_answer} (attempt: {attempt})")
                                else:
                                    logger.warning(f"MMLU multiple choice answer is invalid: '{model_answer}' (attempt: {attempt}), will re-ask")
                            
                            results["mmlu_results"].append({
                                "question": mmlu_question,
                                "choices": mmlu_question_choices,
                                "model_answer": model_answer
                            })
            
            # Save results
            save_result_to_jsonl(results, args.output_file)
            
            # Add content ID to processed set
            processed_ids.add(content_id)
        
        print("\nAll content analysis completed!")
        logger.info("All content analysis completed!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        logger.error(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()