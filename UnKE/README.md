# UnKE: Unified Knowledge Extraction for Omni-modal Agent Memory Framework

## Project Overview

UnKE is a comprehensive framework for knowledge extraction and evaluation within the Omni-modal Agent Memory Framework. This project focuses on training language models to extract knowledge from various sources and evaluating their performance on question-answering tasks.

## Repository Structure

```
UnKE/
├── LLama-factory_pretrain_sft_files/
│   ├── unke_pretrain.jsonl      # Pretraining data in JSONL format
│   └── unke_sft.json            # Supervised fine-tuning data in JSON format
├── calculate_accuracy.py        # Script for evaluating model performance
├── unke_val.py                  # Script for model inference and evaluation
└── README.md                    # This documentation
```

## Key Components

### 1. Model Evaluation (`calculate_accuracy.py`)

This script evaluates the performance of language models on knowledge extraction tasks by:

- Calculating similarity between model-generated answers and reference answers
- Supporting multiple similarity models (all-MiniLM-L6-v2, bert-base-chinese, chinese-roberta-wwm-ext)
- Evaluating performance on different question types:
  - Original questions
  - Paraphrased questions
  - Sub-questions
  - MMLU (Massive Multitask Language Understanding) questions
- Providing detailed performance metrics including average similarity, max/min scores, and accuracy

### 2. Model Inference (`unke_val.py`)

This script runs inference on pre-trained models to generate answers for evaluation by:

- Loading pre-trained models using ModelScope
- Processing different types of questions from input datasets
- Generating answers with retry mechanisms for robustness
- Saving results in JSON format for further analysis

### 3. Training Data

The project includes two types of training data:

- **Pretraining data** (`unke_pretrain.jsonl`): Contains text passages about various individuals and their occupations
- **Supervised fine-tuning data** (`unke_sft.json`): Contains question-answer pairs for training the model on knowledge extraction tasks

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- Transformers
- ModelScope
- tqdm
- NumPy

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd UnKE
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### 1. Running Model Inference

```bash
python unke_val.py --model_path <path-to-model> --data_file <path-to-dataset> --output_file <output-path>
```

**Parameters:**
- `--model_path`: Path to the pre-trained model (default: `./model/Qwen1.5-7B_unke_pretrain_sft`)
- `--data_file`: Path to the dataset JSON file (default: `Path to UnKE dataset /final_data_v3.json`)
- `--output_file`: Path to save the results (default: `./model_results.json`)
- `--max_retries`: Maximum number of retries for API calls (default: 3)
- `--start_idx`: Starting index for processing entries (default: 0)
- `--end_idx`: Ending index for processing entries (default: None, process all)

#### 2. Calculating Accuracy

```bash
python calculate_accuracy.py --results_file <path-to-results> --dataset_file <path-to-dataset> --similarity_model <model-name>
```

**Parameters:**
- `--results_file`: Path to the model results JSON file (default: `./model_results.json`)
- `--dataset_file`: Path to the dataset annotation JSON file (default: `Path to UnKE dataset ./final_data_v3.json`)
- `--similarity_model`: Model to use for calculating similarity (default: `all-MiniLM-L6-v2`)

## Data Format

### Input Dataset Format

The input dataset should be a JSON file containing entries with the following structure:

```json
{
  "id": "unique_id",
  "question": "Original question",
  "answer": "Reference answer",
  "para_question": "Paraphrased question",
  "sub_question": ["Sub-question 1", "Sub-question 2"],
  "sub_answer": ["Sub-answer 1", "Sub-answer 2"],
  "mmlu_questions": ["MMLU question 1"],
  "mmlu_choices": [["Option 0", "Option 1", "Option 2", "Option 3"]],
  "mmlu_answer": ["Correct answer"]
}
```

### Output Results Format

The model results will be saved in the following format:

```json
{
  "results": [
    {
      "id": "unique_id",
      "question": "Original question",
      "answer": "Model's answer",
      "para_question": "Paraphrased question",
      "para_answer": "Model's answer to paraphrased question",
      "sub_questions": [
        {
          "question": "Sub-question",
          "answer": "Model's answer"
        }
      ],
      "mmlu_results": [
        {
          "question": "MMLU question",
          "choices": ["Option 0", "Option 1", "Option 2", "Option 3"],
          "model_answer": "Model's answer (0-3)"
        }
      ]
    }
  ]
}
```

## Training Process

The project uses a two-stage training process:

1. **Pretraining**: Models are pretrained on general knowledge using `unke_pretrain.jsonl`
2. **Supervised Fine-tuning (SFT)**: Models are fine-tuned on specific question-answer pairs using `unke_sft.json`

## Evaluation Metrics

The framework evaluates model performance using:

- **Similarity scores**: Cosine similarity between model answers and reference answers
- **F1 scores**: Harmonic mean of precision and recall for answer similarity
- **Accuracy**: Exact match accuracy for MMLU questions

## Use Cases

This framework can be used for:

- Evaluating language models on knowledge extraction tasks
- Fine-tuning models for improved question-answering performance
- Building components for omni-modal agent memory systems
- Researching methods for knowledge representation and retrieval

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- This project is part of the Omni-modal Agent Memory Framework
- Uses pre-trained models from Hugging Face and ModelScope
- Inspired by research in knowledge extraction and language model evaluation