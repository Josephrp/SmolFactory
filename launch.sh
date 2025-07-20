#!/bin/bash
# Interactive SmolLM3 End-to-End Fine-tuning Pipeline
# This script creates a complete finetuning pipeline with user configuration

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo -e "${PURPLE}🚀 $1${NC}"
}

print_step() {
    echo -e "${CYAN}📋 $1${NC}"
}

# Function to get user input with default value
get_input() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    
    if [ -n "$default" ]; then
        read -p "$prompt [$default]: " input
        if [ -z "$input" ]; then
            input="$default"
        fi
    else
        read -p "$prompt: " input
        while [ -z "$input" ]; do
            print_error "This field is required!"
            read -p "$prompt: " input
        done
    fi
    
    eval "$var_name=\"$input\""
}

# Function to select from options
select_option() {
    local prompt="$1"
    local options=("${@:2}")
    local var_name="${!#}"
    
    echo "$prompt"
    for i in "${!options[@]}"; do
        echo "  $((i+1)). ${options[$i]}"
    done
    
    while true; do
        read -p "Enter your choice (1-${#options[@]}): " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#options[@]}" ]; then
            eval "$var_name=\"${options[$((choice-1))]}\""
            break
        else
            print_error "Invalid choice. Please enter a number between 1 and ${#options[@]}"
        fi
    done
}

# Function to validate HF token
validate_hf_token() {
    local token="$1"
    if [ -z "$token" ]; then
        return 1
    fi
    
    # Test the token
    export HF_TOKEN="$token"
    if huggingface-cli whoami >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to show training configurations
show_training_configs() {
    echo ""
    print_header "Available Training Configurations"
    echo "======================================"
    echo ""
    echo "1. Basic Training (Default)"
    echo "   - Model: SmolLM3-3B"
    echo "   - Dataset: SmolTalk"
    echo "   - Epochs: 3"
    echo "   - Batch Size: 2"
    echo "   - Learning Rate: 5e-6"
    echo ""
    echo "2. H100 Lightweight (Rapid)"
    echo "   - Model: SmolLM3-3B"
    echo "   - Dataset: OpenHermes-FR (80K samples)"
    echo "   - Epochs: 1"
    echo "   - Batch Size: 16"
    echo "   - Learning Rate: 8e-6"
    echo "   - Sequence Length: 8192"
    echo "   - Optimized for H100 rapid training"
    echo ""
    echo "3. A100 Large Scale"
    echo "   - Model: SmolLM3-3B"
    echo "   - Dataset: OpenHermes-FR"
    echo "   - Epochs: 1.3 passes"
    echo "   - Batch Size: 8"
    echo "   - Learning Rate: 5e-6"
    echo "   - Sequence Length: 8192"
    echo ""
    echo "4. Multiple Passes"
    echo "   - Model: SmolLM3-3B"
    echo "   - Dataset: OpenHermes-FR"
    echo "   - Epochs: 4 passes"
    echo "   - Batch Size: 6"
    echo "   - Learning Rate: 3e-6"
    echo "   - Sequence Length: 8192"
    echo ""
    echo "5. Custom Configuration"
    echo "   - User-defined parameters"
    echo ""
}

# Function to get training configuration
get_training_config() {
    local config_type="$1"
    
    case "$config_type" in
        "Basic Training")
            MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
            DATASET_NAME="legmlai/openhermes-fr"
            MAX_EPOCHS=3
            BATCH_SIZE=2
            GRADIENT_ACCUMULATION_STEPS=8
            LEARNING_RATE=5e-6
            MAX_SEQ_LENGTH=4096
            CONFIG_FILE="config/train_smollm3.py"
            ;;
        "H100 Lightweight (Rapid)")
            MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
            DATASET_NAME="legmlai/openhermes-fr"
            MAX_EPOCHS=1
            BATCH_SIZE=16
            GRADIENT_ACCUMULATION_STEPS=4
            LEARNING_RATE=8e-6
            MAX_SEQ_LENGTH=8192
            DATASET_SAMPLE_SIZE=80000
            CONFIG_FILE="config/train_smollm3_h100_lightweight.py"
            ;;
        "A100 Large Scale")
            MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
            DATASET_NAME="legmlai/openhermes-fr"
            MAX_EPOCHS=1
            BATCH_SIZE=8
            GRADIENT_ACCUMULATION_STEPS=16
            LEARNING_RATE=5e-6
            MAX_SEQ_LENGTH=8192
            CONFIG_FILE="config/train_smollm3_openhermes_fr_a100_large.py"
            ;;
        "Multiple Passes")
            MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
            DATASET_NAME="legmlai/openhermes-fr"
            MAX_EPOCHS=4
            BATCH_SIZE=6
            GRADIENT_ACCUMULATION_STEPS=20
            LEARNING_RATE=3e-6
            MAX_SEQ_LENGTH=8192
            CONFIG_FILE="config/train_smollm3_openhermes_fr_a100_multiple_passes.py"
            ;;
        "Custom Configuration")
            get_custom_config
            ;;
    esac
}

# Function to get custom configuration
get_custom_config() {
    print_step "Custom Configuration Setup"
    echo "============================="
    
    get_input "Model name" "HuggingFaceTB/SmolLM3-3B" MODEL_NAME
    get_input "Dataset name" "HuggingFaceTB/smoltalk" DATASET_NAME
    get_input "Number of epochs" "3" MAX_EPOCHS
    get_input "Batch size" "2" BATCH_SIZE
    get_input "Gradient accumulation steps" "8" GRADIENT_ACCUMULATION_STEPS
    get_input "Learning rate" "5e-6" LEARNING_RATE
    get_input "Max sequence length" "4096" MAX_SEQ_LENGTH
    
    # Select config file based on dataset
    if [[ "$DATASET_NAME" == *"openhermes"* ]]; then
        CONFIG_FILE="config/train_smollm3_openhermes_fr.py"
    else
        CONFIG_FILE="config/train_smollm3.py"
    fi
}

# Function to create training configuration file
create_training_config() {
    local config_file="$1"
    
    cat > "$config_file" << EOF
"""
SmolLM3 Training Configuration - Generated by launch.sh
Optimized for: $TRAINING_CONFIG_TYPE
"""

from config.train_smollm3 import SmolLM3Config

config = SmolLM3Config(
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
    dataset_name="$DATASET_NAME",
    dataset_split="train",
    input_field="prompt",
    target_field="completion",
    filter_bad_entries=False,
    bad_entry_field="bad_entry",
    
    # Chat template configuration
    use_chat_template=True,
    chat_template_kwargs={
        "enable_thinking": False,
        "add_generation_prompt": True,
        "no_think_system_message": True
    },
    
    # Trackio monitoring configuration
    enable_tracking=True,
    trackio_url="$TRACKIO_URL",
    trackio_token=None,
    log_artifacts=True,
    log_metrics=True,
    log_config=True,
    experiment_name="$EXPERIMENT_NAME",
    
    # HF Datasets configuration
    dataset_repo="$TRACKIO_DATASET_REPO"
)
EOF
}

# Main script starts here
print_header "SmolLM3 End-to-End Fine-tuning Pipeline"
echo "=============================================="
echo ""

# Step 1: Get user credentials
print_step "Step 1: User Authentication"
echo "================================"

get_input "Hugging Face username" "" HF_USERNAME
get_input "Hugging Face token (get from https://huggingface.co/settings/tokens)" "" HF_TOKEN

# Validate HF token
print_info "Validating Hugging Face token..."
if validate_hf_token "$HF_TOKEN"; then
    print_status "HF token validated successfully"
else
    print_error "Invalid HF token. Please check your token and try again."
    exit 1
fi

# Step 2: Select training configuration
print_step "Step 2: Training Configuration"
echo "=================================="

show_training_configs
select_option "Select training configuration:" "Basic Training" "H100 Lightweight (Rapid)" "A100 Large Scale" "Multiple Passes" "Custom Configuration" TRAINING_CONFIG_TYPE

get_training_config "$TRAINING_CONFIG_TYPE"

# Step 3: Get experiment details
print_step "Step 3: Experiment Details"
echo "=============================="

get_input "Experiment name" "smollm3_finetune_$(date +%Y%m%d_%H%M%S)" EXPERIMENT_NAME
get_input "Model repository name" "$HF_USERNAME/smollm3-finetuned-$(date +%Y%m%d)" REPO_NAME
get_input "Trackio dataset repository" "$HF_USERNAME/trackio-experiments" TRACKIO_DATASET_REPO

# Step 4: Training parameters
print_step "Step 4: Training Parameters"
echo "==============================="

echo "Current configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME"
if [ "$TRAINING_CONFIG_TYPE" = "H100 Lightweight (Rapid)" ]; then
    echo "  Dataset Sample Size: ${DATASET_SAMPLE_SIZE:-80000}"
fi
echo "  Epochs: $MAX_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Sequence Length: $MAX_SEQ_LENGTH"

get_input "Save steps" "500" SAVE_STEPS
get_input "Evaluation steps" "100" EVAL_STEPS
get_input "Logging steps" "10" LOGGING_STEPS

# Step 5: Trackio Space configuration
print_step "Step 5: Trackio Space Configuration"
echo "======================================"

get_input "Trackio Space name" "trackio-monitoring-$(date +%Y%m%d)" TRACKIO_SPACE_NAME
TRACKIO_URL="https://huggingface.co/spaces/$HF_USERNAME/$TRACKIO_SPACE_NAME"

# Step 6: Confirm configuration
print_step "Step 6: Configuration Summary"
echo "================================="

echo ""
echo "📋 Configuration Summary:"
echo "========================"
echo "  User: $HF_USERNAME"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME"
echo "  Training Config: $TRAINING_CONFIG_TYPE"
if [ "$TRAINING_CONFIG_TYPE" = "H100 Lightweight (Rapid)" ]; then
    echo "  Dataset Sample Size: ${DATASET_SAMPLE_SIZE:-80000}"
fi
echo "  Epochs: $MAX_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Model Repo: $REPO_NAME"
echo "  Trackio Space: $TRACKIO_URL"
echo "  HF Dataset: $TRACKIO_DATASET_REPO"
echo ""

read -p "Proceed with this configuration? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    print_info "Configuration cancelled. Exiting."
    exit 0
fi

# Step 7: Environment setup
print_step "Step 7: Environment Setup"
echo "============================"

print_info "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y git curl wget unzip python3-pip python3-venv

print_info "Creating Python virtual environment..."
python3 -m venv smollm3_env
source smollm3_env/bin/activate

print_info "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

print_info "Installing project dependencies..."
pip install -r requirements/requirements_core.txt

print_info "Installing additional dependencies..."
pip install trl>=0.7.0
pip install peft>=0.4.0
pip install accelerate>=0.20.0
pip install huggingface-hub>=0.16.0
pip install datasets>=2.14.0
pip install requests>=2.31.0

# Step 8: Authentication setup
print_step "Step 8: Authentication Setup"
echo "================================"

export HF_TOKEN="$HF_TOKEN"
export TRACKIO_DATASET_REPO="$TRACKIO_DATASET_REPO"
huggingface-cli login --token $HF_TOKEN

# Step 9: Deploy Trackio Space
print_step "Step 9: Deploying Trackio Space"
echo "==================================="

cd scripts/trackio_tonic

# Create deployment script input
cat > deploy_input.txt << EOF
$HF_USERNAME
$TRACKIO_SPACE_NAME
$HF_TOKEN
EOF

# Run deployment script
python deploy_trackio_space.py < deploy_input.txt

print_status "Trackio Space deployed: $TRACKIO_URL"

# Step 10: Setup HF Dataset
print_step "Step 10: Setting up HF Dataset"
echo "=================================="

cd ../dataset_tonic
python setup_hf_dataset.py

# Step 11: Configure Trackio
print_step "Step 11: Configuring Trackio"
echo "================================="

cd ../trackio_tonic
python configure_trackio.py

# Step 12: Create training configuration
print_step "Step 12: Creating Training Configuration"
echo "==========================================="

cd ../..
create_training_config "$CONFIG_FILE"

# Step 13: Download and prepare dataset
print_step "Step 13: Preparing Dataset"
echo "==============================="

python -c "
from datasets import load_dataset
import json
import os
import random

# Load dataset
print('Loading dataset: $DATASET_NAME')
dataset = load_dataset('$DATASET_NAME')

# Create dataset directory
os.makedirs('training_dataset', exist_ok=True)

# Convert to training format
def convert_to_training_format(example):
    # Handle different dataset formats
    if 'prompt' in example and 'completion' in example:
        return {
            'prompt': example['prompt'],
            'completion': example['completion']
        }
    elif 'instruction' in example and 'output' in example:
        return {
            'prompt': example['instruction'],
            'completion': example['output']
        }
    elif 'messages' in example:
        # Handle chat format
        messages = example['messages']
        if len(messages) >= 2:
            return {
                'prompt': messages[0]['content'],
                'completion': messages[1]['content']
            }
    else:
        # Fallback
        return {
            'prompt': str(example.get('input', '')),
            'completion': str(example.get('output', ''))
        }

# Process train split
train_data = []
for example in dataset['train']:
    training_example = convert_to_training_format(example)
    if training_example['prompt'] and training_example['completion']:
        train_data.append(training_example)

# Apply dataset sampling for lightweight configuration
if '$TRAINING_CONFIG_TYPE' == 'H100 Lightweight (Rapid)' and len(train_data) > ${DATASET_SAMPLE_SIZE:-0}:
    print(f'Sampling {${DATASET_SAMPLE_SIZE:-80000}} random samples from {len(train_data)} total samples')
    random.seed(42)  # For reproducibility
    train_data = random.sample(train_data, ${DATASET_SAMPLE_SIZE:-80000})
    print(f'Selected {len(train_data)} samples for lightweight training')

# Process validation split if available
val_data = []
if 'validation' in dataset:
    for example in dataset['validation']:
        training_example = convert_to_training_format(example)
        if training_example['prompt'] and training_example['completion']:
            val_data.append(training_example)

# For lightweight config, also sample validation if it's large
if '$TRAINING_CONFIG_TYPE' == 'H100 Lightweight (Rapid)' and len(val_data) > 1000:
    print(f'Sampling 1000 random validation samples from {len(val_data)} total')
    random.seed(42)  # For reproducibility
    val_data = random.sample(val_data, 1000)

# Save to files
with open('training_dataset/train.json', 'w') as f:
    json.dump(train_data, f, indent=2)

if val_data:
    with open('training_dataset/validation.json', 'w') as f:
        json.dump(val_data, f, indent=2)

print(f'Dataset prepared: {len(train_data)} train samples, {len(val_data)} validation samples')
"

# Step 14: Calculate training parameters
print_step "Step 14: Calculating Training Parameters"
echo "============================================"

TOTAL_SAMPLES=$(python -c "import json; data=json.load(open('training_dataset/train.json')); print(len(data))")
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
STEPS_PER_EPOCH=$((TOTAL_SAMPLES / EFFECTIVE_BATCH_SIZE))
MAX_STEPS=$((STEPS_PER_EPOCH * MAX_EPOCHS))

echo "  Total samples: $TOTAL_SAMPLES"
echo "  Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "  Steps per epoch: $STEPS_PER_EPOCH"
echo "  Total training steps: $MAX_STEPS"

# Step 15: Start training
print_step "Step 15: Starting Training"
echo "=============================="

python src/train.py "$CONFIG_FILE" \
    --dataset_dir training_dataset \
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
    --experiment_name "$EXPERIMENT_NAME" \
    --hf_token "$HF_TOKEN" \
    --dataset_repo "$TRACKIO_DATASET_REPO"

# Step 16: Push model to Hugging Face Hub
print_step "Step 16: Pushing Model to HF Hub"
echo "====================================="

python scripts/model_tonic/push_to_huggingface.py /output-checkpoint "$REPO_NAME" \
    --token "$HF_TOKEN" \
    --trackio-url "$TRACKIO_URL" \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-repo "$TRACKIO_DATASET_REPO"

# Step 17: Test the uploaded model
print_step "Step 17: Testing Uploaded Model"
echo "==================================="

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

# Step 18: Create summary report
print_step "Step 18: Creating Summary Report"
echo "===================================="

cat > training_summary.md << EOF
# SmolLM3 Fine-tuning Summary

## Configuration
- **Model**: $MODEL_NAME
- **Dataset**: $DATASET_NAME
- **Experiment**: $EXPERIMENT_NAME
- **Repository**: $REPO_NAME
- **Trackio Space**: $TRACKIO_URL
- **HF Dataset**: $TRACKIO_DATASET_REPO
- **Training Config**: $TRAINING_CONFIG_TYPE
$(if [ "$TRAINING_CONFIG_TYPE" = "H100 Lightweight (Rapid)" ]; then
echo "- **Dataset Sample Size**: ${DATASET_SAMPLE_SIZE:-80000}"
fi)

## Training Parameters
- **Batch Size**: $BATCH_SIZE
- **Gradient Accumulation**: $GRADIENT_ACCUMULATION_STEPS
- **Learning Rate**: $LEARNING_RATE
- **Max Epochs**: $MAX_EPOCHS
- **Max Steps**: $MAX_STEPS
- **Total Samples**: $TOTAL_SAMPLES
- **Sequence Length**: $MAX_SEQ_LENGTH

## Results
- **Model Repository**: https://huggingface.co/$REPO_NAME
- **Trackio Monitoring**: $TRACKIO_URL
- **Experiment Data**: https://huggingface.co/datasets/$TRACKIO_DATASET_REPO

## Next Steps
1. Monitor training progress in your Trackio Space
2. Check the model repository on Hugging Face Hub
3. Use the model in your applications
4. Share your results with the community

## Files Created
- Training configuration: \`$CONFIG_FILE\`
- Dataset: \`training_dataset/\`
- Model checkpoint: \`/output-checkpoint/\`
- Training logs: \`training.log\`
- Summary report: \`training_summary.md\`
EOF

print_status "Summary report saved to: training_summary.md"

# Final summary
echo ""
print_header "🎉 End-to-End Pipeline Completed Successfully!"
echo "=================================================="
echo ""
echo "📊 Model: https://huggingface.co/$REPO_NAME"
echo "📈 Trackio: $TRACKIO_URL"
echo "📋 Experiment: $EXPERIMENT_NAME"
echo "📊 Dataset: https://huggingface.co/datasets/$TRACKIO_DATASET_REPO"
echo ""
echo "📋 Summary report saved to: training_summary.md"
echo ""
echo "🚀 Next steps:"
echo "1. Monitor training progress in your Trackio Space"
echo "2. Check the model repository on Hugging Face Hub"
echo "3. Use the model in your applications"
echo "4. Share your results with the community"
echo ""
print_status "Pipeline completed successfully!" 