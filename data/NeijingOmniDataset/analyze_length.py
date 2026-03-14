import json

# 分析JSONL文件的text字段长度
def analyze_jsonl(file_path):
    lengths = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if 'text' in data:
                        text = data['text']
                        lengths.append(len(text))
                except json.JSONDecodeError:
                    pass
    return lengths

# 分析JSON文件的instruction和output字段长度
def analyze_json(file_path):
    instruction_lengths = []
    output_lengths = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if 'instruction' in item:
                    instruction_lengths.append(len(item['instruction']))
                if 'output' in item:
                    output_lengths.append(len(item['output']))
    return instruction_lengths, output_lengths

# 输出统计结果
def print_statistics(name, lengths):
    if lengths:
        max_length = max(lengths)
        avg_length = sum(lengths) / len(lengths)
        total_records = len(lengths)
        
        # 按长度分段统计
        bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 350), (350, 400), (400, 450), (450, 500), (500, float('inf'))]
        bin_counts = {f'{low}-{high if high != float("inf") else "+"}': 0 for low, high in bins}
        
        for length in lengths:
            for i, (low, high) in enumerate(bins):
                if low <= length < high:
                    bin_key = f'{low}-{high if high != float("inf") else "+"}'
                    bin_counts[bin_key] += 1
                    break
        
        # 输出结果
        print(f"\n{name}:")
        print(f"总记录数: {total_records}")
        print(f"最大长度: {max_length}")
        print(f"平均长度: {avg_length:.2f}")
        print("按长度分段统计:")
        for bin_key, count in bin_counts.items():
            print(f"{bin_key}: {count} 条")
    else:
        print(f"\n{name}: 文件中没有有效记录")

# 主函数
if __name__ == "__main__":
    # 分析neijing_pretrain.jsonl
    pretrain_file = '/home/cccc/Documents/E/我的论文/Omni-modalAgentMemoryFramework/code/Neijing/LLama-factory_pretrain_sft_files/neijing_pretrain.jsonl'
    pretrain_lengths = analyze_jsonl(pretrain_file)
    print_statistics("neijing_pretrain.jsonl (text字段)", pretrain_lengths)
    
    # 分析neijing_sft.json
    sft_file = '/home/cccc/Documents/E/我的论文/Omni-modalAgentMemoryFramework/code/Neijing/LLama-factory_pretrain_sft_files/neijing_sft.json'
    instruction_lengths, output_lengths = analyze_json(sft_file)
    print_statistics("neijing_sft.json (instruction字段)", instruction_lengths)
    print_statistics("neijing_sft.json (output字段)", output_lengths)