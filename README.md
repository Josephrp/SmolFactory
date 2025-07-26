# SmolLM3 Fine-tuning

This repository provides a complete setup for fine-tuning SmolLM3 models using the FlexAI console, following the nanoGPT structure but adapted for modern transformer models.

## Overview

SmolLM3 is a 3B-parameter transformer decoder model optimized for efficiency, long-context reasoning, and multilingual support. This setup allows you to fine-tune SmolLM3 for various tasks including:

- **Supervised Fine-tuning (SFT)**: Adapt the model for instruction following
- **Direct Preference Optimization (DPO)**: Improve model alignment
- **Long-context fine-tuning**: Support for up to 128k tokens
- **Tool calling**: Fine-tune for function calling capabilities
- **Model Quantization**: Create int8 (GPU) and int4 (CPU) quantized versions

## Quick Start

### 1. Repository Setup

The repository follows the FlexAI console structure with the following key files:

- `train.py`: Main entry point script
- `config/train_smollm3.py`: Default configuration
- `model.py`: Model wrapper and loading
- `data.py`: Dataset handling and preprocessing
- `trainer.py`: Training loop and trainer setup
- `requirements.txt`: Dependencies

### 2. FlexAI Console Configuration

When setting up a Fine Tuning Job in the FlexAI console, use these settings:

#### Basic Configuration
- **Name**: `smollm3-finetune`
- **Cluster**: Your organization's designated cluster
- **Checkpoint**: (Optional) Previous training job checkpoint
- **Node Count**: 1
- **Accelerator Count**: 1-8 (depending on your needs)

#### Repository Settings
- **Repository URL**: `https://github.com/your-username/flexai-finetune`
- **Repository Revision**: `main`

#### Dataset Configuration
- **Datasets**: Your dataset (mounted under `/input`)
- **Mount Directory**: `my_dataset`

#### Entry Point
```
train.py config/train_smollm3.py --dataset_dir=my_dataset --init_from=resume --out_dir=/input-checkpoint --max_iters=1500
```

### 3. Dataset Format

The script supports multiple dataset formats:

#### Chat Format (Recommended)
```json
[
  {
    "messages": [
      {"role": "user", "content": "What is machine learning?"},
      {"role": "assistant", "content": "Machine learning is a subset of AI..."}
    ]
  }
]
```

#### Instruction Format
```json
[
  {
    "instruction": "What is machine learning?",
    "output": "Machine learning is a subset of AI..."
  }
]
```

#### User-Assistant Format
```json
[
  {
    "user": "What is machine learning?",
    "assistant": "Machine learning is a subset of AI..."
  }
]
```

### 4. Configuration Options

The default configuration in `config/train_smollm3.py` includes:

```python
@dataclass
class SmolLM3Config:
    # Model configuration
    model_name: str = "HuggingFaceTB/SmolLM3-3B"
    max_seq_length: int = 4096
    use_flash_attention: bool = True
    
    # Training configuration
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    max_iters: int = 1000
    
    # Mixed precision
    fp16: bool = True
    bf16: bool = False
```

### 5. Command Line Arguments

The `train.py` script accepts various arguments:

```bash
# Basic usage
python train.py config/train_smollm3.py

# With custom parameters
python train.py config/train_smollm3.py \
    --dataset_dir=my_dataset \
    --out_dir=/output-checkpoint \
    --init_from=resume \
    --max_iters=1500 \
    --batch_size=8 \
    --learning_rate=1e-5 \
    --max_seq_length=8192
```

## Advanced Usage

### 1. Custom Configuration

Create a custom configuration file:

```python
# config/my_config.py
from config.train_smollm3 import SmolLM3Config

config = SmolLM3Config(
    model_name="HuggingFaceTB/SmolLM3-3B-Instruct",
    max_seq_length=8192,
    batch_size=2,
    learning_rate=1e-5,
    max_iters=2000,
    use_flash_attention=True,
    fp16=True
)
```

### 2. Long-Context Fine-tuning

For long-context tasks (up to 128k tokens):

```python
config = SmolLM3Config(
    max_seq_length=131072,  # 128k tokens
    model_name="HuggingFaceTB/SmolLM3-3B",
    use_flash_attention=True,
    gradient_checkpointing=True
)
```

### 3. DPO Training

For preference optimization, use the DPO trainer:

```python
from trainer import SmolLM3DPOTrainer

dpo_trainer = SmolLM3DPOTrainer(
    model=model,
    dataset=dataset,
    config=config,
    output_dir="./dpo-output"
)

dpo_trainer.train()
```

### 4. Tool Calling Fine-tuning

Include tool calling examples in your dataset:

```json
[
  {
    "messages": [
      {"role": "user", "content": "What's the weather in New York?"},
      {"role": "assistant", "content": "<tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"location\">New York</parameter>\n</invoke>\n</tool_call>"},
      {"role": "tool", "content": "The weather in New York is 72°F and sunny."},
      {"role": "assistant", "content": "The weather in New York is currently 72°F and sunny."}
    ]
  }
]
```

## Model Variants

SmolLM3 comes in several variants:

- **SmolLM3-3B-Base**: Base model for general fine-tuning
- **SmolLM3-3B**: Instruction-tuned model
- **SmolLM3-3B-Instruct**: Enhanced instruction model
- **Quantized versions**: Available for deployment

## Hardware Requirements

### Minimum Requirements
- **GPU**: 16GB+ VRAM (for 3B model)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space

### Recommended
- **GPU**: A100/H100 or similar
- **RAM**: 64GB+ system memory
- **Storage**: 100GB+ SSD

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce `batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable `gradient_checkpointing`
   - Use `fp16` or `bf16`

2. **Slow Training**
   - Enable `flash_attention`
   - Use mixed precision (`fp16`/`bf16`)
   - Increase `dataloader_num_workers`

3. **Dataset Loading Issues**
   - Check dataset format
   - Ensure proper JSON structure
   - Verify file permissions

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Evaluation

After training, evaluate your model:

```python
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="./output-checkpoint",
    device=0,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7
)

# Test the model
messages = [{"role": "user", "content": "Explain gravity in simple terms."}]
outputs = pipe(messages)
print(outputs[0]["generated_text"][-1]["content"])
```

## Model Quantization

The pipeline includes built-in quantization support using torchao for creating optimized model versions with a unified repository structure:

### Repository Structure

All models (main and quantized) are stored in a single repository:

```
your-username/model-name/
├── README.md (unified model card)
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── int8/ (quantized model for GPU)
└── int4/ (quantized model for CPU)
```

### Quantization Types

- **int8_weight_only**: GPU optimized, ~50% memory reduction
- **int4_weight_only**: CPU optimized, ~75% memory reduction

### Automatic Quantization

When using the interactive pipeline (`launch.sh`), you'll be prompted to create quantized versions after training:

```bash
./launch.sh
# ... training completes ...
# Choose quantization options when prompted
```

### Standalone Quantization

Quantize existing models independently:

```bash
# Quantize and push to HF Hub (same repository)
python scripts/model_tonic/quantize_standalone.py /path/to/model your-username/model-name \
    --quant-type int8_weight_only \
    --token YOUR_HF_TOKEN

# Quantize and save locally
python scripts/model_tonic/quantize_standalone.py /path/to/model your-username/model-name \
    --quant-type int4_weight_only \
    --device cpu \
    --save-only
```

### Loading Quantized Models

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load main model
model = AutoModelForCausalLM.from_pretrained(
    "your-username/model-name",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("your-username/model-name")

# Load int8 quantized model (GPU)
model = AutoModelForCausalLM.from_pretrained(
    "your-username/model-name/int8",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("your-username/model-name/int8")

# Load int4 quantized model (CPU)
model = AutoModelForCausalLM.from_pretrained(
    "your-username/model-name/int4",
    device_map="cpu",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("your-username/model-name/int4")
```

For detailed quantization documentation, see [QUANTIZATION_GUIDE.md](docs/QUANTIZATION_GUIDE.md).

### Unified Model Cards

The system generates comprehensive model cards that include information about all model variants:

- **Single README**: One comprehensive model card for the entire repository
- **Conditional Sections**: Quantized model information appears when available
- **Usage Examples**: Complete examples for all model variants
- **Performance Information**: Memory and speed benefits for each quantization type

For detailed information about the unified model card system, see [UNIFIED_MODEL_CARD_GUIDE.md](docs/UNIFIED_MODEL_CARD_GUIDE.md).

## Deployment

### Using vLLM
```bash
vllm serve ./output-checkpoint --enable-auto-tool-choice
```

### Using llama.cpp
```bash
# Convert to GGUF format
python -m llama_cpp.convert_model ./output-checkpoint --outfile model.gguf
```

## Resources

- [SmolLM3 Blog Post](https://huggingface.co/blog/smollm3)
- [Model Repository](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- [GitHub Repository](https://github.com/huggingface/smollm)
- [SmolTalk Dataset](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)

## License

This project follows the same license as the SmolLM3 model. Please refer to the Hugging Face model page for licensing information. 


{
  "id": "exp_20250718_195852",
  "name": "petit-elle-l-aime-3",
  "description": "SmolLM3 fine-tuning experiment",
  "created_at": "2025-07-18T19:58:52.689087",
  "status": "running",
  "metrics": [],
  "parameters": {},
  "artifacts": [],
  "logs": []
}