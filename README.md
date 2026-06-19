# A Python framework for paper *Keep It Simple: A Memory Framework for Omni-modal Agents Based on Cerebrum-Cerebellum Coordination*. 

This project implements a framework for generating and comparing answers using multiple LLMs (Large Language Models), comparing their outputs, and leveraging Retrieval-Augmented Generation (RAG) for enhanced responses.

## Features

- **Multiple LLM Support**: Load and use different Qwen1.5 models for answer generation
- **Text Similarity Calculation**: Compare generated answers using pre-trained Chinese BERT models
- **Retrieval-Augmented Generation (RAG)**: Enhance answers with reference information when model outputs differ significantly
- **Dataset Processing**: Load and process JSON datasets containing questions and reference answers
- **Result Persistence**: Save all generated answers and similarity scores to CSV files for analysis
- **Model Caching**: Efficiently reuse loaded models to improve performance
- **Reproducible Results**: Configurable random seed for consistent model outputs

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- NumPy
- OpenAI Python Client
- Requests

## Installation

1. Clone or navigate to the project directory:
   ```bash
   cd ./frame
   ```

2. Install the required dependencies:
   ```bash
   pip install torch transformers numpy openai
   ```

## Model Preparation

提前下载以下模型到 `model` 文件夹：

- **Qwen1.5-7B-Chat**
- **Qwen1.5-7B_neijing_sft** (URL: `https://modelscope.cn/models/ctsean/Qwen1.5-7B_neijing_sft/`)
- **chinese-roberta-wwm-ext**
- **all-MiniLM-L6-v2**

## Usage

### Step 1: Generate Fine-tuned Medium-scale LLM (Cerebellum) results

运行以下命令得到 Fine-tuned Medium-scale LLM (Cerebellum) 的结果 $Answer_{ft}$：

```bash
python qwen1.5-7B_result.py --model_path ./model/Qwen1.5-7B_neijing_sft --output_file ./result/Qwen1.5-7B_neijing_sft_results.jsonl
```

### Step 2: Generate Original Medium-scale LLM results

运行以下命令得到 Original Medium-scale LLM 的结果 $Answer_{org}$：

```bash
python qwen1.5-7B_result.py --model_path ./model/Qwen1.5-7B-Chat --output_file ./result/Qwen1.5-7B-Chat_results.jsonl
```

### Step 3: Merge results for Cerebellum-based RAG

运行以下命令合并结果，得到用于 Cerebellum-based RAG 的输入：

```bash
python merge_SFT_ORG.py
```

### Step 4: Generate final Cerebrum results

运行以下命令得到最终结果：

```bash
python cerebrum_result.py
```

### Step 5: Calculate answer similarity

运行以下命令计算最终结果与标注答案的相似度：

```bash
python calculate_answer_similarity.py
```

## Key Components

### Answer Generation

The framework uses Qwen1.5 models to generate answers with deterministic settings (low temperature, greedy decoding) for reproducible results.

### Similarity Calculation

Answers from different models are compared using Chinese RoBERTa embeddings to measure semantic similarity.

### Retrieval-Augmented Generation (RAG)

When model answers have low similarity (< 0.7), the framework uses the fine-tuned model's answer as reference for Qwen3-Max to generate an enhanced response.

### Result Output

Results are saved to CSV files with the following columns:
- `Question Content`
- `Original Answer`
- `Model 1 Answer`
- `Model 2 Answer`
- `chinese-roberta-wwm-ext Similarity`
- `LLM Answer`
- `Timestamp`
- `Data Filename`

## Example Output

Sample CSV entry:

| Question Content | Original Answer | Model 1 Answer | Model 2 Answer | chinese-roberta-wwm-ext Similarity | LLM Answer | Timestamp | Data Filename |
|------------------|----------------|----------------|----------------|-------------------------------------|------------|-----------|---------------|
| What is yin and yang? | Yin and yang are ancient Chinese philosophical concepts... | Yin and yang are basic categories of ancient philosophy... | Yin and yang refer to interconnected and opposing phenomena in the universe... | 0.8523 | Yin and yang are core concepts in ancient Chinese philosophy, referring to interconnected and opposing phenomena... | 2024-02-09 14:30:22 | NeijingOmni-modalDataset.json |

## Acknowledgments

- Qwen1.5 and Qwen3-Max models by Alibaba Cloud
- Chinese RoBERTa model from Hugging Face
- Transformers library for model implementations