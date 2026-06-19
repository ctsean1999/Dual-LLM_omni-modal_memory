# A Python framework for paper *Keep It Simple: A Memory Framework for Omni-modal Agents Based on Cerebrum-Cerebellum Coordination*. 

This project implements a framework for generating and comparing answers using multiple LLMs (Large Language Models), comparing their outputs, and leveraging Retrieval-Augmented Generation (RAG) for enhanced responses.

## Features

- **Multiple LLM Support**: Load and use different Qwen1.5 models for answer generation
- **Text Similarity Calculation**: Compare generated answers using pre-trained Chinese BERT models
- **Retrieval-Augmented Generation (RAG)**: Enhance answers with reference information when model outputs differ significantly
- **Dataset Processing**: Load and process JSON datasets containing questions and reference answers
- **Result Persistence**: Save all generated answers to JSON files for analysis
- **Model Caching**: Efficiently reuse loaded models to improve performance
- **Reproducible Results**: Configurable random seed for consistent model outputs

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- NumPy
- ModelScope
- tqdm
- volcenginesdkarkruntime

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ctsean1999/Dual-LLM_omni-modal_memory.git
   cd Dual-LLM_omni-modal_memory
   ```

2. Install the required dependencies:
   ```bash
   pip install torch transformers numpy modelscope tqdm 
   pip install 'volcengine-python-sdk[ark]' 
   ```

## Model Preparation

Download the following models to the `model` folder in advance:

- **Qwen1.5-7B-Chat**
- **Qwen1.5-7B_neijing_sft** (URL: `https://modelscope.cn/models/ctsean/Qwen1.5-7B_neijing_sft/`)
- **chinese-roberta-wwm-ext**
- **all-MiniLM-L6-v2**

## Usage

### Step 1: Generate Fine-tuned Medium-scale LLM (Cerebellum) results

Run the following command to get the results for Fine-tuned Medium-scale LLM (Cerebellum) $Answer_{ft}$:

```bash
python qwen1.5-7B_result.py --model_path ./model/Qwen1.5-7B_neijing_sft --output_file ./result/Qwen1.5-7B_neijing_sft_results.jsonl
```

### Step 2: Generate Original Medium-scale LLM results

Run the following command to get the results for Original Medium-scale LLM $Answer_{org}$:

```bash
python qwen1.5-7B_result.py --model_path ./model/Qwen1.5-7B-Chat --output_file ./result/Qwen1.5-7B-Chat_results.jsonl
```

### Step 3: Merge results for Cerebellum-based RAG

Run the following command to merge results and get the input for Cerebellum-based RAG:

```bash
python merge_SFT_ORG.py
```

### Step 4: Generate final Cerebrum results

Run the following command to get the final results:

```bash
python cerebrum_result.py
```

### Step 5: Calculate answer similarity

Run the following command to calculate the similarity between the final results and the annotated answers:

```bash
python calculate_answer_similarity.py
```

## Key Components

### Answer Generation

The framework uses Qwen1.5 models to generate answers with deterministic settings (low temperature, greedy decoding) for reproducible results.

### Similarity Calculation

Answers from different models are compared using Chinese RoBERTa embeddings to measure semantic similarity.

### Retrieval-Augmented Generation (RAG)

When model answers have low similarity (< 0.8), the framework uses the fine-tuned model's answer as reference for Large-scale Omni-modal Model (Cerebrum) to generate an enhanced response.

## Acknowledgments

- Qwen1.5 and Qwen3-Max models by Alibaba Cloud
- Chinese RoBERTa model from Hugging Face
- Transformers library for model implementations