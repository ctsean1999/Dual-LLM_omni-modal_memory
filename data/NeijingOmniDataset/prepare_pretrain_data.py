import json

# 读取JSON文件
input_file = './annotation.json'
output_file = './LLama-factory_pretrain_sft_files/data/neijing_pretrain.jsonl'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取video_content和knowledge并保存为JSONL
with open(output_file, 'w', encoding='utf-8') as f:
    for item in data:
        # 处理video_content
        if 'video_content' in item and item['video_content']:
            f.write(json.dumps({'text': item['video_content']}, ensure_ascii=False) + '\n')
        # 处理knowledge
        if 'knowledge' in item and item['knowledge']:
            f.write(json.dumps({'text': item['knowledge']}, ensure_ascii=False) + '\n')

print(f"处理完成，已保存到 {output_file}")