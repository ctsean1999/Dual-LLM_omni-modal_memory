# Fix bug in Python importlib
import sys
import importlib
import gc
import multiprocessing
import json
import torch
import gc
import time
import argparse
from tqdm import tqdm
import transformers
from functools import partial
import fcntl  # For file locking
import re
import numpy as np
import os

# Save original invalidate_caches function
original_invalidate_caches = importlib.invalidate_caches

# Define fixed function
def fixed_invalidate_caches():
    try:
        return original_invalidate_caches()
    except TypeError as e:
        if "missing 1 required positional argument: 'cls'" in str(e):
            # Ignore this error as it's a bug in PyTorch distributed module
            return
        else:
            raise e

# Replace original function
importlib.invalidate_caches = fixed_invalidate_caches

# Batch size
BATCH_SIZE = 10  # Adjust based on each GPU's memory, 4 GPUs in parallel, total batch size equivalent to original 40

def load_model(model_name, gpu_id):
    """Load model to specified GPU, using transformers.pipeline for batch processing"""
    # Ensure each process only uses the specified GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        print(f"GPU {gpu_id}: Starting to load model {model_name}...")
        print(f"GPU {gpu_id}: Using half precision (float16)")
        
        # Create text-generation pipeline
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={
                "dtype": torch.float16,  # Use half precision to reduce memory usage
                # "low_cpu_mem_usage": True,  # Low CPU memory usage
            },
            device_map="auto",  # Automatically allocate device
        )
        print(f"GPU {gpu_id}: Model loaded successfully, Pipeline created")
        print(f"GPU {gpu_id}: Model architecture: {pipeline.model.__class__.__name__}")
        print(f"GPU {gpu_id}: Device allocation: {pipeline.device}")
        return pipeline
    except Exception as e:
        print(f"GPU {gpu_id}: Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_batch(prompts, pipeline, gpu_id, output_file, original_instructions=None, start_id=None):
    """Process a batch of data on the specified GPU and save results immediately"""
    try:
        batch_size = len(prompts)
        print(f"GPU {gpu_id}: Processing batch - Sample count: {batch_size}, Max generation length: 512")
        
        # Print GPU memory usage - in multiprocessing environment, each process can only see its own GPU (device 0)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU {gpu_id}: Memory usage - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
        
        # Process batch
        batch_responses = pipeline(
            prompts,
            max_new_tokens=512,
            use_cache=True,  # Use cache to speed up generation
            truncation=True,  # Enable truncation
            return_full_text=False,  # Only return generated part
            temperature=0.0,
            do_sample=False,
        )
        
        print(f"GPU {gpu_id}: Batch processing completed, generated responses: {len(batch_responses)}")
        
        # Extract results and save to file immediately
        batch_results = []
        for i, response in enumerate(batch_responses):
            try:
                # Process response structure returned by transformers.pipeline
                if isinstance(response, list) and len(response) > 0:
                    # First layer is a list, take the first element
                    first_item = response[0]
                    
                    if isinstance(first_item, dict):
                        # Check if 'generated_text' key exists
                        if 'generated_text' in first_item:
                            generated_text = first_item['generated_text']
                            
                            # Ensure generated_text is a string
                            if isinstance(generated_text, list):
                                # If it's a list, convert to string
                                # First check if elements in the list are strings
                                str_list = []
                                for item in generated_text:
                                    if isinstance(item, str):
                                        str_list.append(item)
                                    else:
                                        str_list.append(str(item))
                                generated_text = ' '.join(str_list)
                            elif not isinstance(generated_text, str):
                                # If it's another type, convert to string
                                generated_text = str(generated_text)
                        else:
                            # If no 'generated_text' key, try to get other possible text fields
                            # Or convert the entire dictionary to string
                            print(f"GPU {gpu_id}: No 'generated_text' key in response dictionary: {list(first_item.keys())}")
                            generated_text = str(first_item)
                    elif isinstance(first_item, str):
                        # If the first element is directly a string
                        generated_text = first_item
                    else:
                        # Other cases, convert to string
                        generated_text = str(first_item)
                else:
                    # Response is not in expected list format, convert to string
                    generated_text = str(response)
                
                # If original instructions are provided, try to extract answer part from generated text
                if original_instructions and i < len(original_instructions):
                    original_instruction = original_instructions[i]
                    # If generated text contains original instruction, only keep the answer part
                    if original_instruction in generated_text:
                        # Extract content after original instruction as answer
                        answer_start = generated_text.find(original_instruction) + len(original_instruction)
                        answer_text = generated_text[answer_start:].strip()
                        batch_results.append(answer_text)
                    else:
                        # Try more flexible way to extract answer
                        # Look for content after "user" role, usually model's answer is after assistant role
                        if 'user' in generated_text and 'assistant' in generated_text:
                            # Find last assistant response
                            last_assistant_idx = generated_text.rfind('assistant')
                            # Extract content after this position
                            answer_start = generated_text.find('content', last_assistant_idx)
                            if answer_start != -1:
                                answer_start = generated_text.find("[", answer_start + 10)  # Skip "content":
                                if answer_start != -1:
                                    # answer_start += 1  # Skip first quote
                                    answer_end = generated_text.find("'", answer_start)
                                    if answer_end != -1:
                                        answer_text = generated_text[answer_start:answer_end]
                                        batch_results.append(answer_text)
                                    else:
                                        batch_results.append(generated_text)
                                else:
                                    batch_results.append(generated_text)
                            else:
                                batch_results.append(generated_text)
                        else:
                            # If no obvious role indicators found, save the entire generated text
                            batch_results.append(generated_text)
                else:
                    batch_results.append(generated_text)
                    
                # Print first 100 characters of first 5 generated results
                if i < 5:
                    preview = generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
                    print(f"GPU {gpu_id}: Sample {i+1} generated result: {preview}")
            except Exception as e:
                print(f"GPU {gpu_id}: Error processing response {i+1}: {e}")
                # Print detailed error information and response structure
                print(f"GPU {gpu_id}: Response structure: {type(response)}, Content: {response}")
                batch_results.append('')
        
        # Use file lock to ensure multiprocess-safe writing
        print(f"GPU {gpu_id}: Preparing to write batch results to file...")
        with open(output_file, 'a', encoding='utf-8') as f:
            # Get file lock
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                # Create jsonl format output with id field
                for i, result in enumerate(batch_results):
                    current_id = start_id + i if start_id is not None else None
                    result_dict = {
                        "id": current_id,
                        "answer": result
                    }
                    json.dump(result_dict, f, ensure_ascii=False)
                    f.write('\n')
                f.flush()  # Ensure data is written to disk
                print(f"GPU {gpu_id}: Batch results successfully written to file")
            finally:
                # Release file lock
                fcntl.flock(f, fcntl.LOCK_UN)
        
        return batch_results
    except Exception as e:
        print(f"GPU {gpu_id}: Error processing batch: {e}")
        import traceback
        traceback.print_exc()
        
        # Even if there's an error, write empty results to keep result count consistent
        error_results = [''] * len(prompts)
        with open(output_file, 'a', encoding='utf-8') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                # Create jsonl format error output with id field
                for i, result in enumerate(error_results):
                    current_id = start_id + i if start_id is not None else None
                    result_dict = {
                        "id": current_id,
                        "answer": result
                    }
                    json.dump(result_dict, f, ensure_ascii=False)
                    f.write('\n')
                f.flush()
                print(f"GPU {gpu_id}: Batch error results written to file")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        
        return error_results


def worker_process(data_chunk, gpu_id, start_index, model_name, output_file):
    """Worker process function, process data chunk and save results in real-time"""
    total_samples = len(data_chunk)
    print(f"GPU {gpu_id}: Starting to process data chunk - Start index: {start_index}, Total samples: {total_samples}")
    
    # Load model only once
    pipeline = load_model(model_name, gpu_id)
    if pipeline is None:
        print(f"GPU {gpu_id}: Model loading failed, data chunk processing failed")
        return False
    
    # Prepare all prompts and corresponding ids
    prompts = []
    record_ids = []
    print(f"GPU {gpu_id}: Preparing prompts...")
    for i, record in enumerate(data_chunk):
        instruction = record.get('instruction', '')
        record_id = record.get('id', start_index + i)  # Get id field, use index if not present
        messages = [
            {"role": "user", "content": instruction}
        ]
        prompts.append({
            'messages': messages,
            'original_instruction': instruction
        })
        record_ids.append(record_id)
        # Print first 100 characters of first 5 prompts
        if i < 5:
            preview = instruction[:100] + "..." if len(instruction) > 100 else instruction
            print(f"GPU {gpu_id}: Prompt {i+1}: {preview}")
    
    print(f"GPU {gpu_id}: Prompt preparation completed, total: {len(prompts)}")
    
    # Process in batches
    num_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"GPU {gpu_id}: Total batches: {num_batches}, Batch size: {BATCH_SIZE}")
    
    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min((batch_idx + 1) * BATCH_SIZE, len(prompts))
        batch_prompts = [p['messages'] for p in prompts[start:end]]
        original_instructions = [p['original_instruction'] for p in prompts[start:end]]
        batch_ids = record_ids[start:end]  # Get current batch ids
        batch_start_id = batch_ids[0] if batch_ids else None
        
        print(f"GPU {gpu_id}: Processing batch {batch_idx + 1}/{num_batches} - Range: {start}~{end-1}, Samples: {end-start}")
        
        # Pass output_file parameter to process_batch to save results in real-time
        process_batch(batch_prompts, pipeline, gpu_id, output_file, original_instructions, batch_start_id)
        
        # Print processing progress
        processed = min((batch_idx + 1) * BATCH_SIZE, len(prompts))
        progress = (processed / len(prompts)) * 100
        print(f"GPU {gpu_id}: Processing progress: {processed}/{len(prompts)} ({progress:.1f}%)")
        
        # Perform small resource cleanup every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"GPU {gpu_id}: Performing resource cleanup every 10 batches...")
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.5)  # Brief sleep
    
    # Clean up resources
    print(f"GPU {gpu_id}: Starting resource cleanup...")
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU {gpu_id}: Resource cleanup completed")
    
    print(f"GPU {gpu_id}: Data chunk processing completed")
    return True


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Multi-GPU model inference for Finding Dory")
    parser.add_argument('--model_name', type=str, default="./model/Qwen1.5-7Bfindingdory_pretrain_sft/", help="Path to the model")
    parser.add_argument('--input_file', type=str, default="./findingdory_validation_file.jsonl", help="Path to input JSONL file")
    parser.add_argument('--output_file', type=str, default="./model_result.jsonl", help="Path to output JSONL file")
    parser.add_argument('--num_gpus', type=int, default=4, help="Number of GPUs to use")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size per GPU")
    
    args = parser.parse_args()
    
    # Use the provided arguments or default values
    model_name = args.model_name
    input_file = args.input_file
    output_file = args.output_file
    num_gpus = args.num_gpus
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    
    # Read JSONL file
    print(f"Main process: Reading input file {input_file}...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError as e:
                    print(f"Main process: Error parsing line {line_num + 1}: {e}")
                    continue
    print(f"Main process: File reading completed, found {len(data)} records")
    
    # data = data[:16]
    # Sort data by id to ensure correct output order
    data.sort(key=lambda x: x.get('id', 0))
    print(f"Main process: Data sorted by id")
    
    # Split data evenly into num_gpus parts
    chunk_size = len(data) // num_gpus
    data_chunks = [
        data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_gpus - 1)
    ]
    # Last chunk contains remaining data
    data_chunks.append(data[(num_gpus - 1) * chunk_size:])
    
    # Calculate start index for each chunk
    start_indices = [i * chunk_size for i in range(num_gpus)]
    
    # Print data sharding information
    print(f"Main process: Data sharding information -")
    for i in range(num_gpus):
        chunk_len = len(data_chunks[i])
        if chunk_len > 0:
            first_id = data_chunks[i][0].get('id', start_indices[i])
            last_id = data_chunks[i][-1].get('id', start_indices[i] + chunk_len - 1)
            print(f"  GPU {i}: Start index {start_indices[i]}, Sample count {chunk_len}, ID range: {first_id}-{last_id}")
        else:
            print(f"  GPU {i}: Start index {start_indices[i]}, Sample count {chunk_len}")
    
    # Create an empty output file
    print(f"Main process: Creating empty output file {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write an empty file, preparing for subsequent append writes
        pass
    
    # Create multiprocessing pool
    print("Main process: Creating multiprocessing pool...")
    with multiprocessing.Pool(processes=num_gpus) as pool:
        # Generate parameter list
        args = []
        for i in range(num_gpus):
            args.append((data_chunks[i], i, start_indices[i], model_name, output_file))
        
        # Process data in parallel
        print("Main process: Starting parallel data processing...")
        print("Main process: Results will be saved to file immediately after each batch is processed...")
        print("Main process: Waiting for all GPU processes to complete...")
        
        # Collect return status of each process (True indicates success)
        process_statuses = pool.starmap(worker_process, args)
    
    # Check status of all processes
    success_count = sum(1 for status in process_statuses if status)
    print(f"Main process: All GPU processes completed - Success: {success_count}, Failed: {num_gpus - success_count}")
    
    # Count lines and id information in output file
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f)
        print(f"Main process: Output file total lines: {line_count}")
        
        # Verify output file id order
        output_ids = []
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if 'id' in record:
                            output_ids.append(record['id'])
                    except json.JSONDecodeError:
                        continue
        
        if output_ids:
            print(f"Main process: Output file ID range: {min(output_ids)}-{max(output_ids)}")
            # Check if ids are in order
            sorted_ids = sorted(output_ids)
            if output_ids == sorted_ids:
                print(f"Main process: Output file ids are in order")
            else:
                print(f"Main process: Warning - Output file ids are not in order")
        
        if line_count != len(data):
            print(f"Main process: Warning - Output line count {line_count} does not match input line count {len(data)}")
        else:
            print(f"Main process: Output line count matches input line count")
    except Exception as e:
        print(f"Main process: Error counting output file lines: {e}")
    
    print(f"Main process: All records processed!")
    print(f"Main process: Results saved to {output_file}")
    print(f"Main process: Total records processed: {len(data)}")
    print(f"Main process: Output format is JSONL, containing id and answer fields")
    
    # Read result file, sort by id, and write back
    print(f"Main process: Reading result file for sorting...")
    try:
        results = []
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        results.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Main process: Error parsing result line: {e}")
                        continue
        
        # Sort by id
        results.sort(key=lambda x: x.get('id', 0))
        print(f"Main process: Result file sorted by id, total {len(results)} records")
        
        # Write back to file
        print(f"Main process: Writing sorted results back to file...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in results:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Main process: Sorted results written back to {output_file}")
        
        # Verify sorting results
        if len(results) > 0:
            ids = [r.get('id', 0) for r in results]
            print(f"Main process: ID range: {min(ids)}-{max(ids)}")
            if ids == sorted(ids):
                print(f"Main process: Result file ids correctly sorted")
            else:
                print(f"Main process: Warning - Result file id sorting may have issues")
                
    except Exception as e:
        print(f"Main process: Error sorting result file: {e}")


if __name__ == "__main__":
    main()