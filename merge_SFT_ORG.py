#!/usr/bin/env python3
import json
import os
import re
import logging
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./log/merge_SFT_ORG.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ANSWER_FT = "./result/Qwen1.5-7B_neijing_sft_results.jsonl"
ANSWER_ORG = "./result/Qwen1.5-7B-Chat_results.jsonl"
OUTPUT_FILE = "./result/answer_ft_org_merged_results_chinese-roberta-wwm-ext_0.8.jsonl"

MODEL_PATH = './model/chinese-roberta-wwm-ext'
# MODEL_PATH = './model/all-MiniLM-L6-v2'
SIMILARITY_THRESHOLD = 0.8

_model_cache = {}

def preprocess_text(text):
    if not text:
        return ""
    text = re.sub(r'[^一-龥a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_model_and_tokenizer():
    if 'model' in _model_cache:
        return _model_cache['model']
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModel.from_pretrained(MODEL_PATH)
        _model_cache['model'] = (tokenizer, model)
        return tokenizer, model
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return None, None

def compute_similarity(tokenizer, model, text1, text2):
    try:
        text1 = preprocess_text(text1)
        text2 = preprocess_text(text2)
        
        if not text1 or not text2:
            return 0.0
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        cand_tokens = tokenizer(text1, return_tensors='pt', padding=True, truncation=True, max_length=512)
        ref_tokens = tokenizer(text2, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        cand_tokens = {k: v.to(device) for k, v in cand_tokens.items()}
        ref_tokens = {k: v.to(device) for k, v in ref_tokens.items()}
        
        with torch.no_grad():
            cand_output = model(**cand_tokens)
            ref_output = model(**ref_tokens)
        
        cand_embeddings = cand_output.last_hidden_state[0]
        ref_embeddings = ref_output.last_hidden_state[0]
        
        cos_sim = torch.nn.functional.cosine_similarity(
            cand_embeddings.unsqueeze(1),
            ref_embeddings.unsqueeze(0),
            dim=2
        )
        
        precision = torch.max(cos_sim, dim=1)[0].mean().item()
        recall = torch.max(cos_sim, dim=0)[0].mean().item()
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    except Exception as e:
        logger.warning(f"计算相似度失败: {e}")
        return 0.0

def load_jsonl(file_path):
    data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        content_id = item.get('content_id')
                        if content_id is not None:
                            data[content_id] = item
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析行失败: {e}")
        logger.info(f"成功从 {file_path} 加载 {len(data)} 条数据")
        return data
    except Exception as e:
        logger.error(f"加载文件失败 {file_path}: {e}")
        return {}

def merge_results():
    try:
        logger.info("加载数据...")
        data1 = load_jsonl(ANSWER_FT)
        data2 = load_jsonl(ANSWER_ORG)
        
        if not data1 or not data2:
            logger.error("无法加载数据文件")
            return
        
        logger.info("加载语义相似度模型...")
        tokenizer, model = load_model_and_tokenizer()
        if tokenizer is None or model is None:
            logger.error("无法加载模型")
            return
        
        stats = {
            'question': {'ft': 0, 'org': 0},
            'paraphrased_question': {'ft': 0, 'org': 0},
            'multihop_qa': {'ft': 0, 'org': 0},
            'mmlu_results': {'ft': 0, 'org': 0}
        }
        
        merged_data = []
        
        common_ids = set(data1.keys()) & set(data2.keys())
        logger.info(f"找到 {len(common_ids)} 个共同的 content_id")
        
        sorted_ids = sorted(common_ids)[:200]
        logger.info(f"只处理前 {len(sorted_ids)} 条数据")
        
        for content_id in tqdm(sorted_ids, desc="处理数据"):
            item1 = data1[content_id]
            item2 = data2[content_id]
            
            merged_item = item1.copy()
            
            if 'answer' in item1 and 'answer' in item2:
                sim = compute_similarity(tokenizer, model, item1['answer'], item2['answer'])
                if sim > SIMILARITY_THRESHOLD:
                    merged_item['answer'] = ""
                    stats['question']['ft'] += 1
                else:
                    stats['question']['org'] += 1
            elif 'answer' in item2:
                stats['question']['org'] += 1
            
            if 'paraphrased_answer' in item1 and 'paraphrased_answer' in item2:
                sim = compute_similarity(tokenizer, model, item1['paraphrased_answer'], item2['paraphrased_answer'])
                if sim > SIMILARITY_THRESHOLD:
                    merged_item['paraphrased_answer'] = ""
                    stats['paraphrased_question']['ft'] += 1
                else:
                    stats['paraphrased_question']['org'] += 1
            elif 'paraphrased_answer' in item2:
                stats['paraphrased_question']['org'] += 1
            
            if 'multihop_qa' in item1 and 'multihop_qa' in item2:
                merged_multihop = []
                max_len = max(len(item1['multihop_qa']), len(item2['multihop_qa']))
                for i in range(max_len):
                    if i < len(item1['multihop_qa']) and i < len(item2['multihop_qa']):
                        qa1 = item1['multihop_qa'][i]
                        qa2 = item2['multihop_qa'][i]
                        merged_qa = qa2.copy()
                        
                        if 'model_answer' in qa1 and 'model_answer' in qa2:
                            sim = compute_similarity(tokenizer, model, qa1['model_answer'], qa2['model_answer'])
                            if sim > SIMILARITY_THRESHOLD:
                                merged_qa['model_answer'] = ""
                                stats['multihop_qa']['ft'] += 1
                            else:
                                stats['multihop_qa']['org'] += 1
                        elif 'model_answer' in qa2:
                            stats['multihop_qa']['org'] += 1
                        
                        merged_multihop.append(merged_qa)
                    elif i < len(item2['multihop_qa']):
                        merged_multihop.append(item2['multihop_qa'][i].copy())
                
                merged_item['multihop_qa'] = merged_multihop
            
            if 'mmlu_results' in item1 and 'mmlu_results' in item2:
                merged_mmlu = []
                max_len = max(len(item1['mmlu_results']), len(item2['mmlu_results']))
                for i in range(max_len):
                    if i < len(item1['mmlu_results']) and i < len(item2['mmlu_results']):
                        mmlu1 = item1['mmlu_results'][i]
                        mmlu2 = item2['mmlu_results'][i]
                        merged_mmlu_item = mmlu2.copy()
                        
                        if 'model_answer' in mmlu1 and 'model_answer' in mmlu2:
                            sim = compute_similarity(tokenizer, model, str(mmlu1['model_answer']), str(mmlu2['model_answer']))
                            if sim > SIMILARITY_THRESHOLD:
                                merged_mmlu_item['model_answer'] = ""
                                stats['mmlu_results']['ft'] += 1
                            else:
                                stats['mmlu_results']['org'] += 1
                        elif 'model_answer' in mmlu2:
                            stats['mmlu_results']['org'] += 1
                        
                        merged_mmlu.append(merged_mmlu_item)
                    elif i < len(item2['mmlu_results']):
                        merged_mmlu.append(item2['mmlu_results'][i].copy())
                
                merged_item['mmlu_results'] = merged_mmlu
            
            merged_data.append(merged_item)
        
        logger.info("保存合并结果...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for item in merged_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"\n合并完成！结果保存到 {OUTPUT_FILE}")
        logger.info("\n统计结果：")
        logger.info(f"question: FT={stats['question']['ft']}, ORG={stats['question']['org']}")
        logger.info(f"paraphrased_question: FT={stats['paraphrased_question']['ft']}, ORG={stats['paraphrased_question']['org']}")
        logger.info(f"multihop_qa: FT={stats['multihop_qa']['ft']}, ORG={stats['multihop_qa']['org']}")
        logger.info(f"mmlu_results: FT={stats['mmlu_results']['ft']}, ORG={stats['mmlu_results']['org']}")
        
        total_ft = stats['question']['ft'] + stats['paraphrased_question']['ft'] + stats['multihop_qa']['ft'] + stats['mmlu_results']['ft']
        total_org = stats['question']['org'] + stats['paraphrased_question']['org'] + stats['multihop_qa']['org'] + stats['mmlu_results']['org']
        logger.info(f"\n总计: FT={total_ft}, ORG={total_org}")
        
    except Exception as e:
        logger.error(f"合并结果失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    merge_results()
