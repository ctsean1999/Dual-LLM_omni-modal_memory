import json
import os

# 原始数据集路径
RAW_DATA_PATH = './annotation.json'
# 输出数据集文件名
OUTPUT_PATH = './LLama-factory_pretrain_sft_files/data/neijing_sft.json'

# 读取原始数据
with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 转换为LLaMA-Factory的alpaca格式
formatted_data = []
for item in raw_data:
    # 使用question和answer字段
    formatted_item = {
        'instruction': item.get('question', ''),
        'input': '',  # 这个数据集中没有额外的输入
        'output': item.get('answer', ''),
        'system': '你是一个精通黄帝内经的中医专家，回答用户关于黄帝内经的问题。'
    }
    formatted_data.append(formatted_item)

# 保存格式化后的数据
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)

print(f"数据集已成功转换并保存到: {OUTPUT_PATH}")
print(f"共处理 {len(formatted_data)} 条数据")