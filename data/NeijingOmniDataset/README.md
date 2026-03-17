# Huangdi Neijing Q&A Evaluation Framework

This project provides a set of tools for evaluating large language models' Q&A capabilities on Huangdi Neijing related knowledge, mainly including two core scripts: `neijing_val.py` and `calculate_accuracy.py`.

Full dataset download link: http://ggrs.ltd:8003/download/neijing%2FNeijingOmni-modalDataset.zip

## Feature Introduction

### 1. neijing_val.py

**Function**: Uses pre-trained large language models to answer Huangdi Neijing related questions and saves the results in JSONL format.

**Supported question types**:
- Original question
- Paraphrased question
- Multi-hop QA
- MMLU multiple choice questions

**Core features**:
- Supports multiple pre-trained models
- Includes retry mechanism and error handling
- Incremental processing, supports resuming from breakpoints
- Detailed logging

### 2. calculate_accuracy.py

**Function**: Calculates the similarity between model answers and original answers to evaluate the quality of model responses.

**Supported evaluation types**:
- Original question answer similarity
- Paraphrased question answer similarity
- Multi-hop question answer similarity
- MMLU multiple choice question accuracy

**Core features**:
- Supports multiple pre-trained models for similarity calculation (all-MiniLM-L6-v2, bert-base-chinese, chinese-roberta-wwm-ext)
- Detailed statistical information output
- Text preprocessing to improve similarity calculation accuracy

## Installation Dependencies

```bash
pip install torch transformers modelscope tqdm
```

## Data Preparation

This project provides two scripts for data preparation:

### 1. prepare_pretrain_data.py

**Function**: Extracts `video_content` and `knowledge` fields from annotated data to generate pre-training data.

**Usage**:

```bash
python prepare_pretrain_data.py
```

**Input**: `./annotation.json`
**Output**: `./LLama-factory_pretrain_sft_files/data/neijing_pretrain.jsonl`

**Output format**: JSONL format, each line contains a `text` field with content from `video_content` or `knowledge`.

### 2. prepare_sft_data.py

**Function**: Extracts `question` and `answer` fields from annotated data and converts them to LLaMA-Factory's alpaca format.

**Usage**:

```bash
python prepare_sft_data.py
```

**Input**: `./annotation.json`
**Output**: `./LLama-factory_pretrain_sft_files/data/neijing_sft.json`

**Output format**: JSON format, containing `instruction`, `input`, `output`, and `system` fields.

## Usage

### 1. Using neijing_val.py to generate model answers

**Basic usage**:

```bash
python neijing_val.py --model_path ./model/Qwen1.5-7B_findingdory_neijing_unke_final --output_file ./model_result.jsonl --data_file ./annotation.json
```

**Parameter description**:
- `--model_path`: Pre-trained model path (default: `./model/Qwen1.5-7B_findingdory_neijing_unke_final`)
- `--output_file`: Output result file path (default: `./model_result.jsonl`)
- `--data_file`: Annotated data file path (default: `./annotation.json`)
- `--max_retries`: Maximum number of retries (default: 3)

**Output format**:
Output is in JSONL format, each line contains a video's answer results, including:
- `content_id`: Video ID
- `question`: Original question
- `answer`: Model's answer to the original question
- `paraphrased_question`: Paraphrased question
- `paraphrased_answer`: Model's answer to the paraphrased question
- `multihop_qa`: Multi-hop questions and their answers
- `mmlu_results`: MMLU multiple choice questions and their answers

Model download URL: https://modelscope.cn/models/ctsean/Qwen1.5-7B_neijing_sft/

### 2. Using calculate_accuracy.py to evaluate answer quality

**Basic usage**:

```bash
python calculate_accuracy.py --results_file ./model_result.jsonl --dataset_file ./annotation.json --similarity_model all-MiniLM-L6-v2
```

**Parameter description**:
- `--results_file`: Model answer results file path (default: `./model_result.jsonl`)
- `--dataset_file`: Annotated data file path (default: `./annotation.json`)
- `--similarity_model`: Model used for calculating similarity (default: `all-MiniLM-L6-v2`, options: `bert-base-chinese`, `chinese-roberta-wwm-ext`)

**Output results**:
Output contains the following statistical information:
- Similarity statistics between original question answers and original answers (average, maximum, minimum)
- Similarity statistics between paraphrased question answers and original answers
- Similarity statistics between multi-hop question answers and expected answers
- Accuracy of MMLU multiple choice questions

## Data Format

### Annotated data format (annotation.json)

```json
[
  {
    "id": "video_id",
    "question": "Original question",
    "answer": "Original answer",
    "paraphrased_question": "Paraphrased question",
    "multihop_qa": [
      {
        "question": "Multi-hop question 1",
        "answer": "Multi-hop question 1 answer"
      },
      {
        "question": "Multi-hop question 2",
        "answer": "Multi-hop question 2 answer"
      }
    ],
    "mmlu_questions": ["MMLU question 1", "MMLU question 2"],
    "mmlu_choices": [["Option A", "Option B", "Option C", "Option D"], ["Option A", "Option B", "Option C", "Option D"]],
    "mmlu_answer": ["0", "1"]
  }
]
```

### Model answer result format (model_result.jsonl)

```json
{
  "content_id": "video_id",
  "question": "Original question",
  "answer": "Model answer",
  "paraphrased_question": "Paraphrased question",
  "paraphrased_answer": "Model's answer to paraphrased question",
  "multihop_qa": [
    {
      "question": "Multi-hop question 1",
      "model_answer": "Model's answer to multi-hop question 1"
    },
    {
      "question": "Multi-hop question 2",
      "model_answer": "Model's answer to multi-hop question 2"
    }
  ],
  "mmlu_results": [
    {
      "question": "MMLU question 1",
      "choices": ["Option A", "Option B", "Option C", "Option D"],
      "model_answer": "0"
    },
    {
      "question": "MMLU question 2",
      "choices": ["Option A", "Option B", "Option C", "Option D"],
      "model_answer": "1"
    }
  ]
}
```

## Examples

### Example 1: Using Qwen1.5-7B model to generate answers

```bash
python neijing_val.py --model_path ./model/Qwen1.5-7B_findingdory_neijing_unke_final --output_file ./model_result.jsonl --data_file ./annotation.json
```

### Example 2: Using all-MiniLM-L6-v2 model to evaluate answer quality

```bash
python calculate_accuracy.py --results_file ./model_result.jsonl --dataset_file ./annotation.json --similarity_model all-MiniLM-L6-v2
```

## Notes

1. Ensure the model path is correct and the model has been downloaded properly
2. Ensure the annotated data format is correct and contains all necessary fields
3. When calculating similarity, ensure the specified similarity model has been downloaded to the `./LLM/` directory
4. For large-scale data, the evaluation process may be time-consuming
5. It is recommended to run in a GPU environment to improve processing speed

## Dependent Models

- **Large language models**: Such as Qwen1.5-7B (placed in the `./model/` directory)
- **Similarity calculation models**:
  - all-MiniLM-L6-v2 (placed in `./LLM/all-MiniLM-L6-v2`)
  - bert-base-chinese (placed in `./LLM/bert-base-chinese`)
  - chinese-roberta-wwm-ext (placed in `./LLM/chinese-roberta-wwm-ext`)

## LLaMA-Factory Configuration Files

This project provides LLaMA-Factory configuration files, located in the `./LLama-factory_pretrain_sft_files/` directory:

- `pretrain.yaml`: Pre-training configuration file
- `qwen1_5_full_sft.yaml`: SFT training configuration file
- `dataset_info.json`: Dataset information configuration file
- `neijing_pretrain.jsonl`: Pre-training data (generated by `prepare_pretrain_data.py`)
- `neijing_sft.json`: SFT training data (generated by `prepare_sft_data.py`)

## Model Directory

- `./model/`: Stores pre-trained models, contains model's README.md file

