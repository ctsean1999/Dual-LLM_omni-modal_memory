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
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('final_data_v3_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize model and tokenizer
def initialize_model(model_name):
    """
    Initialize model and tokenizer
    """
    print(f"Loading model and tokenizer...")
    print(f"Using model path: {model_name}")
    
    try:
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            dtype="auto", 
            device_map="auto"
        )
        print(f"✓ Model loaded successfully")
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer loaded successfully")
        
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
        print(f"Successfully loaded metadata, total {len(metadata)} entries")
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
                        "content": "You are a Q&A assistant responsible for answering users' questions."
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

def save_result_to_json(result, json_file):
    """
    Save analysis result to JSON file
    """
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {json_file}")

def main():
    """
    Main function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze final_data_v3 using Qwen1.5-7B model')
    
    # Model path
    parser.add_argument('--model_path', 
                      default='./model/Qwen1.5-7B_unke_pretrain_sft',
                      help='Path to the pre-trained model')
    
    # Output file path
    parser.add_argument('--output_file', 
                      default='./model_results.json',
                      help='Path to save the analysis results in JSON format')
    
    # Data file path
    parser.add_argument('--data_file', 
                      default='Path to UnKE dataset /final_data_v3.json',
                      help='Path to the annotation data JSON file')
    
    # Maximum number of retries
    parser.add_argument('--max_retries', 
                      type=int, 
                      default=3,
                      help='Maximum number of retries for API calls')
    
    # Start index for processing
    parser.add_argument('--start_idx', 
                      type=int, 
                      default=0,
                      help='Starting index for processing entries')
    
    # End index for processing
    parser.add_argument('--end_idx', 
                      type=int, 
                      default=None,
                      help='Ending index for processing entries (None means process all)')
    
    args = parser.parse_args()
    
    try:
        # Check model path
        if not os.path.exists(args.model_path):
            print(f"Error: Model path does not exist: {args.model_path}")
            logger.error(f"Model path does not exist: {args.model_path}")
            return
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Initialize model and tokenizer
        model, tokenizer = initialize_model(args.model_path)
        
        # Load metadata
        metadata = load_metadata(args.data_file)
        
        # Determine processing range
        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx is not None else len(metadata)
        
        print(f"Processing entries from index {start_idx} to {end_idx}")
        
        # Read processed IDs if output file exists (for resuming)
        processed_ids = set()
        if os.path.exists(args.output_file):
            try:
                with open(args.output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                    if isinstance(existing_results, list):
                        for result in existing_results:
                            if 'id' in result:
                                processed_ids.add(result['id'])
                    elif isinstance(existing_results, dict) and 'results' in existing_results:
                        for result in existing_results['results']:
                            if 'id' in result:
                                processed_ids.add(result['id'])
                print(f"Found {len(processed_ids)} previously processed entries")
            except Exception as e:
                logger.warning(f"Failed to read existing output file: {e}")
        
        # Process entries one by one
        all_results = []
        for idx in tqdm(range(start_idx, end_idx), desc="Processing entries"):
            entry = metadata[idx]
            entry_id = entry.get('id', idx)
            
            # Skip if already processed
            if entry_id in processed_ids:
                print(f"Skipping already processed entry ID: {entry_id}")
                continue
            
            print(f"Processing entry ID: {entry_id} (index: {idx})")
            
            # Initialize result dictionary
            results = {
                "id": entry_id
            }
            
            # 1. Answer original question
            if 'question' in entry and entry['question']:
                results["question"] = entry['question']
                results["answer"] = answer_question(
                    entry['question'], 
                    model=model, 
                    tokenizer=tokenizer, 
                    max_retries=args.max_retries
                )
            
            # 2. Answer paraphrased question
            if 'para_question' in entry and entry['para_question']:
                results["para_question"] = entry['para_question']
                results["para_answer"] = answer_question(
                    entry['para_question'], 
                    model=model, 
                    tokenizer=tokenizer, 
                    max_retries=args.max_retries
                )
            
            # 3. Answer all sub-questions
            if 'sub_question' in entry and isinstance(entry['sub_question'], list):
                results["sub_questions"] = []
                for sub_q in entry['sub_question']:
                    if sub_q:
                        sub_answer = answer_question(
                            sub_q, 
                            model=model, 
                            tokenizer=tokenizer, 
                            max_retries=args.max_retries
                        )
                        results["sub_questions"].append({
                            "question": sub_q,
                            "answer": sub_answer
                        })
                print(f"  Processed {len(results['sub_questions'])} sub-questions")
            
            # 4. Answer MMLU multiple choice questions with choices
            if ('mmlu_questions' in entry and isinstance(entry['mmlu_questions'], list) and
                'mmlu_choices' in entry and isinstance(entry['mmlu_choices'], list)):
                
                results["mmlu_results"] = []
                
                # Ensure we have matching questions and choices
                num_mmlu_questions = len(entry['mmlu_questions'])
                num_mmlu_choices = len(entry['mmlu_choices'])
                
                for i in range(num_mmlu_questions):
                    mmlu_question = entry['mmlu_questions'][i]
                    
                    # Get corresponding choices (handle case where choices might be missing)
                    mmlu_choices = []
                    if i < num_mmlu_choices:
                        mmlu_choices = entry['mmlu_choices'][i]
                    
                    if mmlu_question:
                        # Ensure choices format is correct (should be a list of 4 options)
                        if not (isinstance(mmlu_choices, list) and len(mmlu_choices) == 4):
                            if not isinstance(mmlu_choices, list):
                                mmlu_choices = []
                            # Pad or truncate choices to exactly 4
                            mmlu_choices = mmlu_choices[:4]
                            while len(mmlu_choices) < 4:
                                mmlu_choices.append("")
                        
                        # Answer MMLU multiple choice question with validation
                        valid_answer = False
                        attempt = 0
                        model_answer = None
                        
                        while not valid_answer and attempt < args.max_retries:
                            attempt += 1
                            model_answer = answer_question(
                                mmlu_question, 
                                choices=mmlu_choices, 
                                model=model, 
                                tokenizer=tokenizer, 
                                max_retries=1  # Only 1 retry here since we have our own retry loop
                            )
                            
                            # Validate if answer is one of 0, 1, 2, 3
                            if model_answer and model_answer.strip() in ['0', '1', '2', '3']:
                                valid_answer = True
                                print(f"MMLU answer valid: {model_answer} (attempt: {attempt})")
                            else:
                                print(f"MMLU answer invalid: '{model_answer}' (attempt: {attempt}), retrying")
                        
                        results["mmlu_results"].append({
                            "question": mmlu_question,
                            "choices": mmlu_choices,
                            "model_answer": model_answer
                        })
                
                print(f"  Processed {len(results['mmlu_results'])} MMLU questions")
            
            # Add result to all results
            all_results.append(results)
            processed_ids.add(entry_id)
            
            # Save intermediate results every 10 entries
            if len(all_results) % 10 == 0:
                save_result_to_json({"results": all_results}, args.output_file)
                print(f"Saved intermediate results ({len(all_results)} entries)")
        
        # Save final results
        if all_results:
            save_result_to_json({"results": all_results}, args.output_file)
            print(f"Final results saved with {len(all_results)} entries")
        
        print("\nAll processing completed!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        logger.error(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
