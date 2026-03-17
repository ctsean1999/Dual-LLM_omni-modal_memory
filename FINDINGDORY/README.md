# FINDINGDORY

## Project Introduction

FINDINGDORY is a fine-tuning and evaluation framework based on the Qwen1.5-7B model, focusing on improving model performance on specific tasks. This project uses LLama-factory for model pre-training and fine-tuning, and provides a complete evaluation process.

## Project Structure

```
FINDINGDORY/
├── LLama-factory_pretrain_sft_files/  # LLama-factory training related files
│   ├── config/                        # Training configuration files
│   │   ├── pretrain.yaml              # Pre-training configuration
│   │   ├── sft.yaml                   # Fine-tuning configuration
│   │   └── merge.yaml                 # Model merging configuration
│   ├── data/                          # Data configuration
│   │   └── dataset_info.json          # Dataset information
│   ├── findingdory_api_pretrain.jsonl # Pre-training data
│   └── findingdory_api_sft.zip        # Fine-tuning data (needs to be unzipped)
├── model/                             # Model directory
│   └── Qwen1.5-7Bfindingdory_pretrain_sft/ # Trained model
├── calculate_accuracy.py              # Accuracy calculation script
├── fast_calculate_accuracy.py         # Fast accuracy calculation script
├── findingdory_val_multi_GPU.py       # Multi-GPU testing script
├── findingdory_validation_file.jsonl  # Test data
├── model_result.jsonl                 # Model evaluation results
└── validation_set_golden_answer.txt   # Validation set golden answers
```

## Environment Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+
- LLama-factory
- Other dependencies (see installation steps below)

## Installation

1. Clone the project

```bash
git clone <project-url>
cd FINDINGDORY
```

2. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers tqdm numpy scipy
pip install -r requirements.txt
```

3. Install LLama-factory

```bash
pip install git+https://github.com/hiyouga/LLaMA-Factory.git
```

## Dependencies

The project requires the following Python libraries:

- **Core dependencies:**
  - `torch` - PyTorch for deep learning
  - `transformers` - Hugging Face Transformers library for model loading and inference
  - `numpy` - Numerical computing
  - `tqdm` - Progress bar
  - `argparse` - Command-line argument parsing
  - `json` - JSON file handling
  - `multiprocessing` - Parallel processing
  - `gc` - Garbage collection
  - `time` - Time utilities
  - `re` - Regular expressions
  - `os` - Operating system utilities

- **Optional dependencies:**
  - `scipy` - For linear assignment problem in accuracy calculation
  - `torchvision` - For vision-related functionalities
  - `torchaudio` - For audio-related functionalities


## Usage

### Step 1: Fine-tune Qwen1.5-7B using LLama-factory

1. Prepare data
   - Unzip the `LLama-factory_pretrain_sft_files/findingdory_api_sft.zip` file

2. Run pre-training

```bash
llama-factory-cli train --config LLama-factory_pretrain_sft_files/config/pretrain.yaml
```

3. Run fine-tuning

```bash
llama-factory-cli train --config LLama-factory_pretrain_sft_files/config/sft.yaml
```

4. Merge models

```bash
llama-factory-cli export --config LLama-factory_pretrain_sft_files/config/merge.yaml
```

After training, the model will be saved in the `./model/Qwen1.5-7Bfindingdory_pretrain_sft` directory.

### Step 2: Test the model

Run the multi-GPU testing script to evaluate model performance using test data:

```bash
python findingdory_val_multi_GPU.py
```
Model download URL: https://modelscope.cn/models/ctsean/Qwen1.5-7B_findingdory_sft
Test results will be saved in the `model_result.jsonl` file.

### Step 3: Calculate accuracy

Run the accuracy calculation script to evaluate the model's performance:

```bash
python calculate_accuracy.py
```

## Data Description

- `findingdory_api_pretrain.jsonl`: Pre-training data
- `findingdory_api_sft.json`: Fine-tuning data (after unzipping)
- `findingdory_validation_file.jsonl`: Test data
- `validation_set_golden_answer.txt`: Validation set golden answers

## Model Description

- `Qwen1.5-7Bfindingdory_pretrain_sft`: Qwen1.5-7B model fine-tuned using LLama-factory

