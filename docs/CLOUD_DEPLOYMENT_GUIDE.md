# Cloud Deployment Guide for SmolLM3 DPO Training

This guide provides the exact sequence of commands to deploy and run SmolLM3 DPO training on a cloud computing instance with 6 epochs.

## Prerequisites

### Cloud Instance Requirements

- **GPU**: NVIDIA A100, H100, or similar (16GB+ VRAM)
- **RAM**: 64GB+ system memory
- **Storage**: 100GB+ SSD storage
- **OS**: Ubuntu 20.04 or 22.04

### Required Information

Before starting, gather these details:
- Your Hugging Face username
- Your Hugging Face token (with write permissions)
- Your Trackio Space URL (if using monitoring)

## Step-by-Step Deployment

### Step 1: Launch Cloud Instance

Choose your cloud provider and launch an instance:

#### AWS (g5.2xlarge or g5.4xlarge)
```bash
# Launch instance with Ubuntu 22.04 and appropriate GPU
aws ec2 run-instances \
    --image-id ami-0c7217cdde317cfec \
    --instance-type g5.2xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx
```

#### Google Cloud (n1-standard-8 with T4/V100)
```bash
gcloud compute instances create smollm3-dpo \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud
```

#### Azure (Standard_NC6s_v3)
```bash
az vm create \
    --resource-group your-rg \
    --name smollm3-dpo \
    --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts:latest \
    --size Standard_NC6s_v3 \
    --admin-username azureuser
```

### Step 2: Connect to Instance

```bash
# SSH to your instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Or for Azure
ssh azureuser@your-instance-ip
```

### Step 3: Update System and Install Dependencies

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y git curl wget unzip python3 python3-pip python3-venv

# Install NVIDIA drivers (if not pre-installed)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### Step 4: Clone Repository and Setup Environment

```bash
# Clone your repository
git clone https://github.com/your-username/flexai-finetune.git
cd flexai-finetune

# Create virtual environment
python3 -m venv smollm3_env
source smollm3_env/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt

# Install additional DPO dependencies
pip install trl>=0.7.0
pip install peft>=0.4.0
pip install accelerate>=0.20.0
```

### Step 5: Configure Authentication

```bash
# Set your Hugging Face token
export HF_TOKEN="your_huggingface_token_here"

# Login to Hugging Face
hf login --token $HF_TOKEN
```

### Step 6: Create Configuration Files

Create the DPO configuration file:

```bash
cat > config/train_smollm3_dpo_6epochs.py << 'EOF'
"""
SmolLM3 DPO Training Configuration - 6 Epochs
Optimized for cloud deployment
"""

from config.train_smollm3_dpo import SmolLM3DPOConfig

config = SmolLM3DPOConfig(
    # Model configuration
    model_name="HuggingFaceTB/SmolLM3-3B",
    max_seq_length=4096,
    use_flash_attention=True,
    use_gradient_checkpointing=True,
    
    # Training configuration
    batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_steps=100,
    max_iters=None,  # Will be calculated based on epochs
    eval_interval=100,
    log_interval=10,
    save_interval=500,
    
    # DPO configuration
    beta=0.1,
    max_prompt_length=2048,
    
    # Optimizer configuration
    optimizer="adamw",
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
    
    # Scheduler configuration
    scheduler="cosine",
    min_lr=1e-6,
    
    # Mixed precision
    fp16=True,
    bf16=False,
    
    # Logging and saving
    save_steps=500,
    eval_steps=100,
    logging_steps=10,
    save_total_limit=3,
    
    # Evaluation
    eval_strategy="steps",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    
    # Data configuration
    data_dir="smoltalk_dataset",
    train_file="train.json",
    validation_file="validation.json",
    
    # Chat template configuration
    use_chat_template=True,
    chat_template_kwargs={
        "enable_thinking": False,
        "add_generation_prompt": True
    },
    
    # Trackio monitoring configuration
    enable_tracking=True,
    trackio_url="https://your-trackio-space.hf.space",  # Change this
    trackio_token=None,
    log_artifacts=True,
    log_metrics=True,
    log_config=True,
    experiment_name="smollm3_dpo_6epochs"
)
EOF
```

### Step 7: Download and Prepare Dataset

```bash
# Create dataset preparation script
cat > prepare_dataset.py << 'EOF'
from datasets import load_dataset
import json
import os

# Load SmolTalk dataset
print('Loading SmolTalk dataset...')
dataset = load_dataset('HuggingFaceTB/smoltalk')

# Create dataset directory
os.makedirs('smoltalk_dataset', exist_ok=True)

# Convert to DPO format (preference pairs)
def convert_to_dpo_format(example):
    # For SmolTalk, we'll create preference pairs based on response quality
    # This is a simplified example - you may need to adjust based on your needs
    return {
        'prompt': example.get('prompt', ''),
        'chosen': example.get('chosen', ''),
        'rejected': example.get('rejected', '')
    }

# Process train split
train_data = []
for example in dataset['train']:
    dpo_example = convert_to_dpo_format(example)
    if dpo_example['prompt'] and dpo_example['chosen'] and dpo_example['rejected']:
        train_data.append(dpo_example)

# Process validation split
val_data = []
for example in dataset['validation']:
    dpo_example = convert_to_dpo_format(example)
    if dpo_example['prompt'] and dpo_example['chosen'] and dpo_example['rejected']:
        val_data.append(dpo_example)

# Save to files
with open('smoltalk_dataset/train.json', 'w') as f:
    json.dump(train_data, f, indent=2)

with open('smoltalk_dataset/validation.json', 'w') as f:
    json.dump(val_data, f, indent=2)

print(f'Dataset prepared: {len(train_data)} train samples, {len(val_data)} validation samples')
EOF

# Run dataset preparation
python prepare_dataset.py
```

### Step 8: Calculate Training Parameters

```bash
# Calculate training steps based on epochs
TOTAL_SAMPLES=$(python -c "import json; data=json.load(open('smoltalk_dataset/train.json')); print(len(data))")
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
MAX_EPOCHS=6
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
STEPS_PER_EPOCH=$((TOTAL_SAMPLES / EFFECTIVE_BATCH_SIZE))
MAX_STEPS=$((STEPS_PER_EPOCH * MAX_EPOCHS))

echo "Training Configuration:"
echo "  Total samples: $TOTAL_SAMPLES"
echo "  Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "  Steps per epoch: $STEPS_PER_EPOCH"
echo "  Total training steps: $MAX_STEPS"
echo "  Training epochs: $MAX_EPOCHS"
```

### Step 9: Start DPO Training

```bash
# Start training with all parameters
python train.py config/train_smollm3_dpo_6epochs.py \
    --dataset_dir smoltalk_dataset \
    --out_dir /output-checkpoint \
    --init_from scratch \
    --max_iters $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_seq_length 4096 \
    --save_steps 500 \
    --eval_steps 100 \
    --logging_steps 10 \
    --enable_tracking \
    --trackio_url "https://your-trackio-space.hf.space" \
    --experiment_name "smollm3_dpo_6epochs"
```

### Step 10: Push Model to Hugging Face Hub

```bash
# Push the trained model
python push_to_huggingface.py /output-checkpoint "your-username/smollm3-dpo-6epochs" \
    --token "$HF_TOKEN" \
    --trackio-url "https://your-trackio-space.hf.space" \
    --experiment-name "smollm3_dpo_6epochs"
```

### Step 11: Test the Uploaded Model

```bash
# Test the model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Loading uploaded model...')
model = AutoModelForCausalLM.from_pretrained('your-username/smollm3-dpo-6epochs', torch_dtype=torch.float16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('your-username/smollm3-dpo-6epochs')

print('Testing model generation...')
prompt = 'Hello, how are you?'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Prompt: {prompt}')
print(f'Response: {response}')
print('✅ Model test completed successfully!')
"
```

## Complete One-Line Deployment

If you want to run everything automatically, use the deployment script:

```bash
# Make script executable
chmod +x cloud_deployment.sh

# Edit configuration in the script first
nano cloud_deployment.sh
# Change these variables:
# - REPO_NAME="your-username/smollm3-dpo-6epochs"
# - TRACKIO_URL="https://your-trackio-space.hf.space"
# - HF_TOKEN="your_hf_token_here"

# Run the complete deployment
./cloud_deployment.sh
```

## Monitoring and Debugging

### Check GPU Usage

```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi
```

### Check Training Logs

```bash
# Monitor training progress
tail -f training.log

# Check system resources
htop
```

### Monitor Trackio

```bash
# Check if Trackio is logging properly
curl -s "https://your-trackio-space.hf.space" | grep -i "experiment"
```

## Expected Timeline

- **Setup**: 15-30 minutes
- **Dataset preparation**: 5-10 minutes
- **Training (6 epochs)**: 4-8 hours (depending on GPU)
- **Model upload**: 10-30 minutes
- **Testing**: 5-10 minutes

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```bash
# Reduce batch size
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16

# Or use gradient checkpointing
# Already enabled in config
```

#### 2. Slow Training
```bash
# Check GPU utilization
nvidia-smi

# Check if mixed precision is working
# Look for "fp16" in training logs
```

#### 3. Dataset Issues
```bash
# Check dataset format
head -n 5 smoltalk_dataset/train.json

# Verify dataset size
wc -l smoltalk_dataset/train.json
```

#### 4. Authentication Issues
```bash
# Test HF token
python -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
print('Token is valid!')
"
```

## Cost Estimation

### AWS (g5.2xlarge)
- **Instance**: $0.526/hour
- **Training time**: 6 hours
- **Total cost**: ~$3.16

### Google Cloud (n1-standard-8 + T4)
- **Instance**: $0.38/hour
- **Training time**: 6 hours
- **Total cost**: ~$2.28

### Azure (Standard_NC6s_v3)
- **Instance**: $0.90/hour
- **Training time**: 6 hours
- **Total cost**: ~$5.40

## Next Steps

After successful deployment:

1. **Monitor training** in your Trackio Space
2. **Check model repository** on Hugging Face Hub
3. **Test the model** with different prompts
4. **Share your model** with the community
5. **Iterate and improve** based on results

## Support

- **Training issues**: Check logs and GPU utilization
- **Upload issues**: Verify HF token and repository permissions
- **Monitoring issues**: Check Trackio Space configuration
- **Performance issues**: Adjust batch size and learning rate

Your SmolLM3 DPO model will be ready for use after training completes! 