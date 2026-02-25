# A Python framework for paper *Keep It Simple: A Memory Framework for Omni-modal Agents Based on Cerebrum-Cerebellum Coordination*. （Code is being organized and will be updated to this repository gradually in the near future.）

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

## Configuration

### Environment Variables

Set up the required API key for accessing Qwen3-Max via Dashscope:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

### Model Paths

The framework uses local models stored in specific paths. Ensure these models are available:

- **Qwen1.5 Models**:
  - Base model: `./LLM/Qwen1.5-7B-Chat`
  - Fine-tuned model: `./model/qwen1_5_7b_pretrain96epoch_merged_lora12epoch_merged`

- **BERT Model**:
  - Chinese RoBERTa: `./LLM/chinese-roberta-wwm-ext`

### Dataset Paths

The framework processes JSON datasets. The default path is:
```
/home/ccc/Documents/myCode/lifelong/myTest/frame/data/NeijingClipsMultimodalDataset339.json
```

You can modify this path in the `main()` function.

## Usage

1. Ensure all required models and dependencies are installed and configured

2. Run the main script:
   ```bash
   python frame.py
   ```

3. The script will:
   - Load the specified dataset
   - Generate answers using the configured Qwen1.5 models
   - Calculate similarity between generated answers
   - Use RAG with Qwen3-Max if similarities are below the threshold (0.7)
   - Save all results to a CSV file in the `result` directory

## Project Structure

```
.
├── frame.py              # Main script containing all functionalities
├── data/                 # Dataset directory
│   └── NeijingOmni-modalDataset.json  # Sample dataset
├── model/                # Fine-tuned model directory
│   └── qwen1_5_7b_pretrain20epoch_merged_lora1200epoch_merged  # Fine-tuned Qwen model
├── result/               # Output CSV directory
└── README.md             # This file
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

## Customization

### Adding New Models

To add new models, update the `model_paths` list in the `main()` function:

```python
model_paths = [
    '/path/to/your/model1',
    '/path/to/your/model2'
]
```

### Adjusting Similarity Threshold

Modify the threshold in the main processing loop:

```python
if similarity < 0.7 and len(model_answers) >= 2:
    # Add reference material
```

### Changing API Parameters

Adjust parameters for the Qwen3-Max API call in `get_model_answer_with_rag()`:

```python
response = client.chat.completions.create(
    model="qwen3-max",
    messages=messages,
    temperature=0.6,
    # Add/modify parameters here
)
```

## Acknowledgments

- Qwen1.5 and Qwen3-Max models by Alibaba Cloud
- Chinese RoBERTa model from Hugging Face
- Transformers library for model implementations