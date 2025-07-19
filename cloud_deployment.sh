#!/bin/bash
# Cloud Deployment Script for SmolLM3 DPO Training
# This script sets up a cloud instance for training and uploading to Hugging Face

set -e  # Exit on any error

echo "🚀 Starting SmolLM3 DPO Cloud Deployment"
echo "=========================================="

# Configuration
MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
DATASET_NAME="HuggingFaceTB/smoltalk"
EXPERIMENT_NAME="smollm3_dpo_6epochs"
REPO_NAME="your-username/smollm3-dpo-6epochs"  # Change this to your username
TRACKIO_URL="https://your-trackio-space.hf.space"  # Change this to your Trackio Space URL
HF_TOKEN="your_hf_token_here"  # Change this to your HF token

# Training Configuration
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=5e-6
MAX_EPOCHS=6
MAX_SEQ_LENGTH=4096
SAVE_STEPS=500
EVAL_STEPS=100
LOGGING_STEPS=10

echo "📋 Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Repository: $REPO_NAME"
echo "  Epochs: $MAX_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"

# Step 1: Update system and install dependencies
echo ""
echo "🔧 Step 1: Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y git curl wget unzip

# Step 2: Install Python and pip
echo ""
echo "🐍 Step 2: Installing Python dependencies..."
sudo apt-get install -y python3 python3-pip python3-venv

# Step 3: Create virtual environment
echo ""
echo "📦 Step 3: Setting up Python virtual environment..."
python3 -m venv smollm3_env
source smollm3_env/bin/activate

# Step 4: Install PyTorch and CUDA
echo ""
echo "🔥 Step 4: Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 5: Install project dependencies
echo ""
echo "📚 Step 5: Installing project dependencies..."
pip install -r requirements.txt

# Step 6: Install additional dependencies for DPO
echo ""
echo "🎯 Step 6: Installing DPO-specific dependencies..."
pip install trl>=0.7.0
pip install peft>=0.4.0
pip install accelerate>=0.20.0

# Step 7: Set up Hugging Face token
echo ""
echo "🔑 Step 7: Setting up Hugging Face authentication..."
export HF_TOKEN="$HF_TOKEN"
huggingface-cli login --token $HF_TOKEN

# Step 8: Create DPO configuration
echo ""
echo "⚙️ Step 8: Creating DPO configuration..."
cat > config/train_smollm3_dpo_6epochs.py << EOF
"""
SmolLM3 DPO Training Configuration - 6 Epochs
Optimized for cloud deployment
"""

from config.train_smollm3_dpo import SmolLM3DPOConfig

config = SmolLM3DPOConfig(
    # Model configuration
    model_name="$MODEL_NAME",
    max_seq_length=$MAX_SEQ_LENGTH,
    use_flash_attention=True,
    use_gradient_checkpointing=True,
    
    # Training configuration
    batch_size=$BATCH_SIZE,
    gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS,
    learning_rate=$LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=100,
    max_iters=None,  # Will be calculated based on epochs
    eval_interval=100,
    log_interval=10,
    save_interval=500,
    
    # DPO configuration
    beta=0.1,
    max_prompt_length=$((MAX_SEQ_LENGTH // 2)),
    
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
    save_steps=$SAVE_STEPS,
    eval_steps=$EVAL_STEPS,
    logging_steps=$LOGGING_STEPS,
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
    trackio_url="$TRACKIO_URL",
    trackio_token=None,
    log_artifacts=True,
    log_metrics=True,
    log_config=True,
    experiment_name="$EXPERIMENT_NAME"
)
EOF

# Step 9: Download and prepare dataset
echo ""
echo "📊 Step 9: Downloading and preparing dataset..."
python -c "
from datasets import load_dataset
import json
import os

# Load SmolTalk dataset
print('Loading SmolTalk dataset...')
dataset = load_dataset('$DATASET_NAME')

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
"

# Step 10: Calculate training steps based on epochs
echo ""
echo "📈 Step 10: Calculating training parameters..."
TOTAL_SAMPLES=$(python -c "import json; data=json.load(open('smoltalk_dataset/train.json')); print(len(data))")
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
STEPS_PER_EPOCH=$((TOTAL_SAMPLES / EFFECTIVE_BATCH_SIZE))
MAX_STEPS=$((STEPS_PER_EPOCH * MAX_EPOCHS))

echo "  Total samples: $TOTAL_SAMPLES"
echo "  Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "  Steps per epoch: $STEPS_PER_EPOCH"
echo "  Total training steps: $MAX_STEPS"

# Step 11: Start DPO training
echo ""
echo "🎯 Step 11: Starting DPO training..."
python train.py config/train_smollm3_dpo_6epochs.py \
    --dataset_dir smoltalk_dataset \
    --out_dir /output-checkpoint \
    --init_from scratch \
    --max_iters $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --enable_tracking \
    --trackio_url "$TRACKIO_URL" \
    --experiment_name "$EXPERIMENT_NAME"

# Step 12: Push model to Hugging Face Hub
echo ""
echo "📤 Step 12: Pushing model to Hugging Face Hub..."
python push_to_huggingface.py /output-checkpoint "$REPO_NAME" \
    --token "$HF_TOKEN" \
    --trackio-url "$TRACKIO_URL" \
    --experiment-name "$EXPERIMENT_NAME"

# Step 13: Test the uploaded model
echo ""
echo "🧪 Step 13: Testing uploaded model..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Loading uploaded model...')
model = AutoModelForCausalLM.from_pretrained('$REPO_NAME', torch_dtype=torch.float16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('$REPO_NAME')

print('Testing model generation...')
prompt = 'Hello, how are you?'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Prompt: {prompt}')
print(f'Response: {response}')
print('✅ Model test completed successfully!')
"

echo ""
echo "🎉 Deployment completed successfully!"
echo "====================================="
echo "📊 Model: https://huggingface.co/$REPO_NAME"
echo "📈 Trackio: $TRACKIO_URL"
echo "📋 Experiment: $EXPERIMENT_NAME"
echo ""
echo "Next steps:"
echo "1. Monitor training progress in your Trackio Space"
echo "2. Check the model repository on Hugging Face Hub"
echo "3. Use the model in your applications" 