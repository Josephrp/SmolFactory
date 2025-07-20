# Push to Hugging Face Hub Guide

This guide explains how to use the `push_to_huggingface.py` script to upload your trained SmolLM3 models and results to Hugging Face Hub.

## Features

- ✅ **Automatic Repository Creation** - Creates HF repositories automatically
- ✅ **Model Validation** - Validates required model files before upload
- ✅ **Comprehensive Model Cards** - Generates detailed model documentation
- ✅ **Training Results Upload** - Uploads logs, configs, and results
- ✅ **Trackio Integration** - Logs push actions to your monitoring system
- ✅ **Private/Public Repositories** - Support for both private and public models

## Prerequisites

### 1. Install Dependencies

```bash
pip install huggingface_hub
```

### 2. Set Up Hugging Face Token

```bash
# Option 1: Environment variable
export HF_TOKEN="your_huggingface_token_here"

# Option 2: Use --token argument
python push_to_huggingface.py model_path repo_name --token "your_token"
```

### 3. Get Your Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "model-upload")
4. Select "Write" permissions
5. Copy the token

## Basic Usage

### Simple Model Push

```bash
python push_to_huggingface.py /path/to/model username/model-name
```

### Push with Custom Token

```bash
python push_to_huggingface.py /path/to/model username/model-name \
    --token "hf_your_token_here"
```

### Push Private Model

```bash
python push_to_huggingface.py /path/to/model username/model-name \
    --private
```

### Push with Trackio Integration

```bash
python push_to_huggingface.py /path/to/model username/model-name \
    --trackio-url "https://your-space.hf.space" \
    --experiment-name "my_experiment"
```

## Complete Workflow Example

### 1. Train Your Model

```bash
python train.py config/train_smollm3.py \
    --dataset_dir my_dataset \
    --enable_tracking \
    --trackio_url "https://your-space.hf.space" \
    --experiment_name "smollm3_finetune_v1"
```

### 2. Push to Hugging Face Hub

```bash
python push_to_huggingface.py /output-checkpoint username/smollm3-finetuned \
    --trackio-url "https://your-space.hf.space" \
    --experiment-name "smollm3_finetune_v1"
```

### 3. Use Your Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your uploaded model
model = AutoModelForCausalLM.from_pretrained("username/smollm3-finetuned")
tokenizer = AutoTokenizer.from_pretrained("username/smollm3-finetuned")

# Generate text
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Repository Structure

After pushing, your repository will contain:

```
username/model-name/
├── README.md                    # Auto-generated model card
├── config.json                  # Model configuration
├── pytorch_model.bin           # Model weights
├── tokenizer.json              # Tokenizer configuration
├── tokenizer_config.json       # Tokenizer settings
├── special_tokens_map.json     # Special tokens
├── training_results/           # Training artifacts
│   ├── train_results.json
│   ├── eval_results.json
│   ├── training_config.json
│   └── training.log
└── .gitattributes             # Git attributes
```

## Model Card Features

The script automatically generates comprehensive model cards including:

- **Model Details**: Base model, fine-tuning method, size
- **Training Configuration**: All training parameters
- **Training Results**: Loss, accuracy, steps, time
- **Usage Examples**: Code snippets for loading and using
- **Performance Metrics**: Training and validation metrics
- **Hardware Information**: GPU/CPU used for training

## Advanced Usage

### Custom Repository Names

```bash
# Public repository
python push_to_huggingface.py /model myusername/smollm3-chatbot

# Private repository
python push_to_huggingface.py /model myusername/smollm3-private --private
```

### Integration with Training Pipeline

```bash
#!/bin/bash
# Complete training and push workflow

# 1. Train the model
python train.py config/train_smollm3.py \
    --dataset_dir my_dataset \
    --enable_tracking \
    --trackio_url "https://your-space.hf.space" \
    --experiment_name "smollm3_v1"

# 2. Push to Hugging Face Hub
python push_to_huggingface.py /output-checkpoint myusername/smollm3-v1 \
    --trackio-url "https://your-space.hf.space" \
    --experiment-name "smollm3_v1"

# 3. Test the model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('myusername/smollm3-v1')
tokenizer = AutoTokenizer.from_pretrained('myusername/smollm3-v1')
print('Model loaded successfully!')
"
```

### Batch Processing Multiple Models

```bash
#!/bin/bash
# Push multiple models

models=(
    "smollm3-baseline"
    "smollm3-high-lr"
    "smollm3-dpo"
)

for model in "${models[@]}"; do
    echo "Pushing $model..."
    python push_to_huggingface.py "/models/$model" "username/$model"
done
```

## Error Handling

### Common Issues and Solutions

#### 1. Missing Model Files

**Error**: `❌ Missing required files: ['config.json', 'pytorch_model.bin']`

**Solution**: Ensure your model directory contains all required files:
- `config.json`
- `pytorch_model.bin`
- `tokenizer.json`
- `tokenizer_config.json`

#### 2. Authentication Issues

**Error**: `❌ Failed to create repository: 401 Client Error`

**Solution**: 
- Check your HF token is valid
- Ensure token has write permissions
- Verify username in repository name matches your account

#### 3. Repository Already Exists

**Error**: `Repository already exists`

**Solution**: The script handles this automatically with `exist_ok=True`, but you can:
- Use a different repository name
- Delete the existing repository first
- Use version numbers: `username/model-v2`

#### 4. Large File Upload Issues

**Error**: `Upload failed for large files`

**Solution**:
- Check your internet connection
- Use Git LFS for large files
- Consider splitting large models

## Trackio Integration

### Logging Push Actions

When using Trackio integration, the script logs:

- **Push Action**: Repository creation and file uploads
- **Model Metadata**: Size, configuration, results
- **Repository Info**: Name, privacy settings, URL
- **Training Results**: Loss, accuracy, steps

### Viewing Push Logs

1. Go to your Trackio Space
2. Navigate to the "View Experiments" tab
3. Find your experiment
4. Check the metrics for push-related actions

## Security Best Practices

### Token Management

```bash
# Use environment variables (recommended)
export HF_TOKEN="your_token_here"
python push_to_huggingface.py model repo

# Don't hardcode tokens in scripts
# ❌ Bad: python push_to_huggingface.py model repo --token "hf_xxx"
```

### Private Models

```bash
# For sensitive models, use private repositories
python push_to_huggingface.py model username/private-model --private
```

### Repository Naming

```bash
# Use descriptive names
python push_to_huggingface.py model username/smollm3-chatbot-v1

# Include version numbers
python push_to_huggingface.py model username/smollm3-v2.0
```

## Performance Optimization

### Large Models

For models > 5GB:

```bash
# Use Git LFS for large files
git lfs install
git lfs track "*.bin"

# Consider splitting models
python push_to_huggingface.py model username/model-large --private
```

### Upload Speed

```bash
# Use stable internet connection
# Consider uploading during off-peak hours
# Use private repositories for faster uploads
```

## Troubleshooting

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python push_to_huggingface.py model repo
```

### Validate Model Files

```bash
# Check model structure before pushing
ls -la /path/to/model/
# Should contain: config.json, pytorch_model.bin, tokenizer.json, etc.
```

### Test Repository Access

```bash
# Test your HF token
python -c "
from huggingface_hub import HfApi
api = HfApi(token='your_token')
print('Token is valid!')
"
```

## Integration Examples

### With CI/CD Pipeline

```yaml
# .github/workflows/train-and-push.yml
name: Train and Push Model

on:
  push:
    branches: [main]

jobs:
  train-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Train Model
        run: |
          python train.py config/train_smollm3.py
      
      - name: Push to HF Hub
        run: |
          python push_to_huggingface.py /output username/model-${{ github.run_number }}
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

### With Docker

```dockerfile
# Dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "push_to_huggingface.py", "/model", "username/model"]
```

## Support and Resources

### Documentation

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Model Cards Guide](https://huggingface.co/docs/hub/model-cards)

### Community

- [Hugging Face Forums](https://discuss.huggingface.co/)
- [GitHub Issues](https://github.com/huggingface/huggingface_hub/issues)

### Examples

- [Model Repository Examples](https://huggingface.co/models?search=smollm3)
- [Fine-tuned Models](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)

## Conclusion

The `push_to_huggingface.py` script provides a complete solution for:

- ✅ **Easy Model Deployment** - One command to push models
- ✅ **Professional Documentation** - Auto-generated model cards
- ✅ **Training Artifacts** - Complete experiment tracking
- ✅ **Integration Ready** - Works with CI/CD and monitoring
- ✅ **Security Focused** - Proper token and privacy management

Start sharing your fine-tuned SmolLM3 models with the community! 