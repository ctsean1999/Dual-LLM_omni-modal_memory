#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import logging
import asyncio
from tqdm import tqdm
from volcenginesdkarkruntime import AsyncArk

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./log/cerebrum_result.log')
    ]
)
logger = logging.getLogger(__name__)


INPUT_FILE = "./result/answer_ft_org_merged_results_chinese-roberta-wwm-ext_0.8.jsonl"
OUTPUT_FILE = "./result/cerebrum_result.jsonl"

MAX_RETRIES = 3
MODEL_NAME = "ark-code-latest"
# MODEL_NAME = "deepseek-v4-flash"

client = AsyncArk(
    # base_url='https://api.deepseek.com',
    # api_key= os.getenv("DEEPSEEK_API_KEY")
    # base_url='https://ark.cn-beijing.volces.com/api/plan/v3',
    # api_key=os.getenv('VOLCENGINE_AGENT_PLAN_API_KEY')
    base_url='https://ark.cn-beijing.volces.com/api/coding/v3',
    api_key= os.getenv("ARK_API_KEY")
)

async def call_model_with_context(prompt, reference_answer, model=MODEL_NAME):
    try:
        full_prompt = f"参考文档：{reference_answer}\n\n问题：{prompt}"
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"调用模型 (尝试 {attempt + 1}/{MAX_RETRIES})")
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": full_prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                
                response_text = None
                if response and response.choices:
                    response_text = response.choices[0].message.content
                
                if response_text:
                    logger.info(f"模型响应: {response_text}...")
                    return response_text.strip()
                
            except Exception as e:
                logger.error(f"调用模型失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    backoff_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"等待 {backoff_time:.2f} 秒后重试...")
                    await asyncio.sleep(backoff_time)
        
        return "调用模型失败"
    except Exception as e:
        logger.error(f"调用模型发生错误: {e}")
        return f"调用模型发生错误: {str(e)}"

async def call_model_for_choice(prompt, choices, reference_answer, model=MODEL_NAME):
    try:
        if reference_answer and len(reference_answer) > 0:
            full_prompt = f"参考文档：{reference_answer}\n\n问题：{prompt}\n\n选项：\n"
        else:
            full_prompt = f"问题：{prompt}\n\n选项：\n"
        for i, choice in enumerate(choices):
            full_prompt += f"{i}. {choice}\n"
        full_prompt += "\n请严格只输出0、1、2、3中的一个数字作为答案，不要输出其他任何内容。"
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"调用模型进行选择题 (尝试 {attempt + 1}/{MAX_RETRIES})")
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": full_prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                
                response_text = None
                if response and response.choices:
                    response_text = response.choices[0].message.content
                
                if response_text:
                    logger.info(f"选择题模型响应: {response_text}")
                    return response_text.strip()
                
            except Exception as e:
                logger.error(f"调用选择题模型失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    backoff_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"等待 {backoff_time:.2f} 秒后重试...")
                    await asyncio.sleep(backoff_time)
        
        return "0"
    except Exception as e:
        logger.error(f"调用选择题模型发生错误: {e}")
        return "0"

async def process_item(item):
    try:
        reference_answer = item.get('answer', '')
        
        result = {
            "content_id": item.get('content_id', 0),
            "question": item.get('question', ''),
            "answer": reference_answer,
            "paraphrased_question": item.get('paraphrased_question', ''),
            "paraphrased_answer": item.get('paraphrased_answer', '')
        }
        
        if result["question"]:
            logger.info(f"处理原始问题: {result['question'][:100]}...")
            model_answer = await call_model_with_context(result["question"], reference_answer, MODEL_NAME)
            result["answer"] = reference_answer
            result["model_answer"] = model_answer
        
        if result["paraphrased_question"]:
            logger.info(f"处理改写问题: {result['paraphrased_question'][:100]}...")
            paraphrased_model_answer = await call_model_with_context(result["paraphrased_question"], reference_answer, MODEL_NAME)
            result["paraphrased_answer"] = item.get('paraphrased_answer', '')
            result["paraphrased_model_answer"] = paraphrased_model_answer
        
        multihop_qa = item.get('multihop_qa', [])
        if multihop_qa and isinstance(multihop_qa, list):
            result["multihop_qa"] = []
            for hop_item in multihop_qa:
                hop_question = hop_item.get('question', '')
                hop_answer = hop_item.get('model_answer', '') if hop_item else ''
                
                if hop_question:
                    logger.info(f"处理多跳问题: {hop_question[:100]}...")
                    model_hop_answer = await call_model_with_context(hop_question, reference_answer, MODEL_NAME)
                    result["multihop_qa"].append({
                        "question": hop_question,
                        "expected_answer": hop_answer,
                        "model_answer": model_hop_answer
                    })
        
        mmlu_results = item.get('mmlu_results', [])
        if mmlu_results and isinstance(mmlu_results, list):
            result["mmlu_results"] = []
            for mmlu_item in mmlu_results:
                mmlu_question = mmlu_item.get('question', '')
                mmlu_choices = mmlu_item.get('choices', [])
                
                if mmlu_question:
                    logger.info(f"处理MMLU问题: {mmlu_question[:100]}...")
                    model_mmlu_answer = await call_model_for_choice(mmlu_question, mmlu_choices, reference_answer, MODEL_NAME)
                    result["mmlu_results"].append({
                        "question": mmlu_question,
                        "choices": mmlu_choices,
                        "model_answer": model_mmlu_answer
                    })
        
        return result
    except Exception as e:
        logger.error(f"处理条目失败: {e}")
        return item

def save_result_to_jsonl(result, jsonl_file):
    with open(jsonl_file, 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')
    logger.info(f"结果已保存到 {jsonl_file}")

def load_data_from_jsonl(file_path, limit=50):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析行 {i+1} 失败: {e}")
        logger.info(f"成功加载 {len(data)} 条数据")
        return data
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return []

async def main_async():
    try:
        data = load_data_from_jsonl(INPUT_FILE, limit=50)
        
        if not data:
            print("没有数据需要处理")
            return
        
        if os.path.exists(OUTPUT_FILE):
            logger.info(f"清空输出文件: {OUTPUT_FILE}")
            os.remove(OUTPUT_FILE)
        
        for i, item in enumerate(tqdm(data, desc="处理数据")):
            logger.info(f"正在处理第 {i+1}/{len(data)} 条数据，content_id: {item.get('content_id', 'N/A')}")
            
            result = await process_item(item)
            
            save_result_to_jsonl(result, OUTPUT_FILE)
        
        print(f"\n处理完成！共处理 {len(data)} 条数据，结果保存在 {OUTPUT_FILE}")
        logger.info(f"处理完成！共处理 {len(data)} 条数据")
        
    except Exception as e:
        print(f"发生错误: {e}")
        logger.error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()