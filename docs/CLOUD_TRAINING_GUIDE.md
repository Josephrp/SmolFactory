# Cloud Training Guide for OpenHermes-FR Dataset

This guide provides step-by-step instructions for training SmolLM3 models on cloud instances using the [legmlai/openhermes-fr](https://huggingface.co/datasets/legmlai/openhermes-fr) dataset.

## Overview

The OpenHermes-FR dataset contains 799,875 French instruction-response pairs, perfect for fine-tuning SmolLM3 models for French language tasks. This guide covers:

- ✅ **Cloud Instance Setup** - Complete environment configuration
- ✅ **Dataset Integration** - Automatic loading and filtering
- ✅ **Training Configuration** - Optimized for French instruction tuning
- ✅ **Monitoring Integration** - Trackio experiment tracking
- ✅ **Model Deployment** - Push to Hugging Face Hub

## Dataset Information

### Schema
```json
{
  "prompt": "Explique la différence entre la photosynthèse C3 et C4.",
  "accepted_completion": "La photosynthèse C3 utilise… (réponse détaillée)",
  "bad_prompt_detected": false,
  "bad_response_detected": false,
  "bad_entry": false
}
```

### Key Features
- **Size**: 799,875 examples (~1.4GB)
- **Language**: 100% French
- **Quality**: GPT-4o generated responses with automatic filtering
- **License**: ODC-BY 1.0

## Cloud Instance Setup

### 1. Choose Your Cloud Provider

#### **AWS EC2 (Recommended)**
```bash
# Launch instance with GPU
# Recommended: g4dn.xlarge or g5.xlarge
# AMI: Deep Learning AMI (Ubuntu 20.04)
```

#### **Google Cloud Platform**
```bash
# Launch instance with GPU
# Recommended: n1-standard-4 with Tesla T4 or V100
```

#### **Azure**
```bash
# Launch instance with GPU
# Recommended: Standard_NC6s_v3 or Standard_NC12s_v3
```

### 2. Instance Specifications

#### **Minimum Requirements**
- **GPU**: 16GB+ VRAM (Tesla T4, V100, or A100)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ SSD
- **CPU**: 8+ cores

#### **Recommended Specifications**
- **GPU**: A100 (40GB) or H100 (80GB)
- **RAM**: 64GB+ system memory
- **Storage**: 200GB+ NVMe SSD
- **CPU**: 16+ cores

### 3. Environment Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install CUDA (if not pre-installed)
# Follow NVIDIA CUDA installation guide for your GPU

# Install Python dependencies
sudo apt install python3-pip python3-venv git -y

# Create virtual environment
python3 -m venv smollm3_env
source smollm3_env/bin/activate

# Clone repository
git clone <your-repo-url>
cd <your-repo-directory>

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for cloud training
pip install accelerate transformers datasets huggingface_hub
```

## Training Configuration

### 1. Use the OpenHermes-FR Config

The repository includes a specialized configuration for the OpenHermes-FR dataset:

```bash
python train.py config/train_smollm3_openhermes_fr.py \
    --enable_tracking \
    --trackio_url "https://your-space.hf.space" \
    --experiment_name "smollm3_fr_openhermes_v1"
```

### 2. Configuration Details

The `config/train_smollm3_openhermes_fr.py` includes:

#### **Dataset Configuration**
```python
dataset_name: str = "legmlai/openhermes-fr"
dataset_split: str = "train"
input_field: str = "prompt"
target_field: str = "accepted_completion"
filter_bad_entries: bool = True
bad_entry_field: str = "bad_entry"
```

#### **Training Optimization**
```python
batch_size: int = 2  # Reduced for French text (longer sequences)
gradient_accumulation_steps: int = 8  # Maintains effective batch size
learning_rate: float = 1e-5  # Lower for instruction tuning
max_iters: int = 2000  # More iterations for large dataset
```

#### **Monitoring Integration**
```python
enable_tracking: bool = True
experiment_name: str = "smollm3_openhermes_fr"
```

## Training Commands

### Basic Training
```bash
python train.py config/train_smollm3_openhermes_fr.py
```

### Training with Monitoring
```bash
python train.py config/train_smollm3_openhermes_fr.py \
    --enable_tracking \
    --trackio_url "https://your-trackio-space.hf.space" \
    --experiment_name "smollm3_fr_openhermes_v1"
```

### Training with Custom Parameters
```bash
python train.py config/train_smollm3_openhermes_fr.py \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --max_iters 3000 \
    --enable_tracking \
    --trackio_url "https://your-trackio-space.hf.space" \
    --experiment_name "smollm3_fr_high_lr"
```

### Training with Checkpoint Resume
```bash
python train.py config/train_smollm3_openhermes_fr.py \
    --init_from resume \
    --enable_tracking \
    --trackio_url "https://your-trackio-space.hf.space" \
    --experiment_name "smollm3_fr_resume"
```

## Dataset Processing

### Automatic Filtering

The training script automatically:
- ✅ **Loads** the OpenHermes-FR dataset from Hugging Face
- ✅ **Filters** out bad entries (`bad_entry = true`)
- ✅ **Splits** data into train/validation/test (98/1/1)
- ✅ **Formats** prompts and completions for instruction tuning

### Manual Dataset Inspection

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("legmlai/openhermes-fr")

# Check dataset info
print(f"Dataset size: {len(dataset['train'])}")
print(f"Sample columns: {dataset['train'].column_names}")

# Check filtering
bad_entries = dataset['train'].filter(lambda x: x['bad_entry'])
print(f"Bad entries: {len(bad_entries)}")

# Sample data
sample = dataset['train'][0]
print(f"Prompt: {sample['prompt']}")
print(f"Completion: {sample['accepted_completion']}")
```

## Monitoring and Tracking

### Trackio Integration

The training automatically logs:
- **Training metrics**: Loss, accuracy, learning rate
- **System metrics**: GPU memory, CPU usage
- **Dataset info**: Size, filtering statistics
- **Model checkpoints**: Regular saves with metadata

### View Training Progress

1. **Trackio Space**: Visit your Trackio Space URL
2. **Experiment Details**: Check the "View Experiments" tab
3. **Metrics**: Monitor loss curves and system usage
4. **Logs**: Download training logs for analysis

## Model Deployment

### Push to Hugging Face Hub

After training, deploy your model:

```bash
python push_to_huggingface.py /output-checkpoint username/smollm3-fr-openhermes \
    --trackio-url "https://your-trackio-space.hf.space" \
    --experiment-name "smollm3_fr_openhermes_v1"
```

### Use Your Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
model = AutoModelForCausalLM.from_pretrained("username/smollm3-fr-openhermes")
tokenizer = AutoTokenizer.from_pretrained("username/smollm3-fr-openhermes")

# Generate French text
prompt = "Expliquez le concept de l'intelligence artificielle."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Performance Optimization

### GPU Memory Management

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Optimize for your GPU
# For 16GB VRAM: batch_size=2, gradient_accumulation_steps=8
# For 24GB VRAM: batch_size=4, gradient_accumulation_steps=4
# For 40GB+ VRAM: batch_size=8, gradient_accumulation_steps=2
```

### Training Speed

```bash
# Use mixed precision (enabled by default)
fp16: bool = True

# Enable gradient checkpointing (enabled by default)
use_gradient_checkpointing: bool = True

# Use flash attention (enabled by default)
use_flash_attention: bool = True
```

## Troubleshooting

### Common Issues

#### 1. **Out of Memory (OOM)**
```bash
# Reduce batch size
python train.py config/train_smollm3_openhermes_fr.py --batch_size 1

# Increase gradient accumulation
# Edit config: gradient_accumulation_steps = 16
```

#### 2. **Slow Training**
```bash
# Check GPU utilization
nvidia-smi

# Verify data loading
# Check if dataset is cached locally
```

#### 3. **Dataset Loading Issues**
```bash
# Clear cache
rm -rf ~/.cache/huggingface/

# Check internet connection
# Verify dataset name: "legmlai/openhermes-fr"
```

#### 4. **Monitoring Connection Issues**
```bash
# Test Trackio connection
curl -I https://your-trackio-space.hf.space

# Check token permissions
# Verify experiment name format
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python train.py config/train_smollm3_openhermes_fr.py
```

## Cost Optimization

### Cloud Provider Tips

#### **AWS EC2**
- Use Spot Instances for cost savings
- Monitor usage with CloudWatch
- Use appropriate instance types

#### **Google Cloud Platform**
- Use Preemptible VMs for non-critical training
- Monitor with Cloud Monitoring
- Use committed use discounts

#### **Azure**
- Use Spot VMs for cost optimization
- Monitor with Azure Monitor
- Use reserved instances for long training

### Training Time Estimates

| GPU Type | Batch Size | Estimated Time |
|----------|------------|----------------|
| Tesla T4 (16GB) | 2 | 8-12 hours |
| V100 (32GB) | 4 | 4-6 hours |
| A100 (40GB) | 8 | 2-3 hours |
| H100 (80GB) | 16 | 1-2 hours |

## Security Best Practices

### Token Management
```bash
# Use environment variables
export HF_TOKEN="your_token_here"
export TRACKIO_TOKEN="your_trackio_token"

# Don't hardcode in scripts
# Use IAM roles when possible
```

### Data Privacy
```bash
# Use private repositories for sensitive models
python push_to_huggingface.py model username/private-model --private

# Secure your cloud instance
# Use VPC and security groups
```

## Complete Workflow Example

### 1. Setup Cloud Instance
```bash
# Launch GPU instance
# Install dependencies
git clone <your-repo>
cd <your-repo>
pip install -r requirements.txt
```

### 2. Train Model
```bash
python train.py config/train_smollm3_openhermes_fr.py \
    --enable_tracking \
    --trackio_url "https://your-space.hf.space" \
    --experiment_name "smollm3_fr_v1"
```

### 3. Deploy Model
```bash
python push_to_huggingface.py /output-checkpoint username/smollm3-fr-v1 \
    --trackio-url "https://your-space.hf.space" \
    --experiment-name "smollm3_fr_v1"
```

### 4. Test Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("username/smollm3-fr-v1")
tokenizer = AutoTokenizer.from_pretrained("username/smollm3-fr-v1")

# Test French generation
prompt = "Qu'est-ce que l'apprentissage automatique?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Support and Resources

### Documentation
- [OpenHermes-FR Dataset](https://huggingface.co/datasets/legmlai/openhermes-fr)
- [SmolLM3 Model](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- [Trackio Monitoring](https://github.com/Josephrp/trackio)

### Community
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

### Examples
- [French Language Models](https://huggingface.co/models?search=french)
- [Instruction Tuned Models](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)

## Conclusion

This guide provides everything needed to train SmolLM3 models on the OpenHermes-FR dataset in the cloud:

- ✅ **Complete Setup** - From cloud instance to model deployment
- ✅ **Optimized Configuration** - Tailored for French instruction tuning
- ✅ **Monitoring Integration** - Trackio experiment tracking
- ✅ **Cost Optimization** - Tips for efficient cloud usage
- ✅ **Troubleshooting** - Solutions for common issues

Start training your French language model today! 