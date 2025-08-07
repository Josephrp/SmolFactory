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

# Function to get secure token input (hidden with stars)
get_secure_token_input() {
    local prompt="$1"
    local var_name="$2"
    local token_type="$3"
    
    echo -n "$prompt: "
    # Use -s flag to hide input, -r to not interpret backslashes
    read -s -r input
    echo  # Add newline after hidden input
    
    # Validate that input is not empty
    while [ -z "$input" ]; do
        print_error "Token is required!"
        echo -n "$prompt: "
        read -s -r input
        echo
    done
    
    # Store the token
    eval "$var_name=\"$input\""
    
    # Show confirmation with stars
    local masked_token="${input:0:4}****${input: -4}"
    print_status "$token_type token added: $masked_token"
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

# Function to validate HF token and get username
validate_hf_token_and_get_username() {
    local token="$1"
    if [ -z "$token" ]; then
        return 1
    fi
    
    # Use Python script for validation
    local result
    if result=$(python3 scripts/validate_hf_token.py "$token" 2>/dev/null); then
        # Parse JSON result using a more robust approach
        local success=$(echo "$result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('success', False))
except:
    print('False')
")
        local username=$(echo "$result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('username', ''))
except:
    print('')
")
        local error=$(echo "$result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('error', 'Unknown error'))
except:
    print('Failed to parse response')
")
        
        if [ "$success" = "True" ] && [ -n "$username" ]; then
            HF_USERNAME="$username"
            return 0
        else
            print_error "Token validation failed: $error"
            return 1
        fi
    else
        print_error "Failed to run token validation script. Make sure huggingface_hub is installed."
        return 1
    fi
}

# Function to show training configurations
show_training_configs() {
    echo ""
    print_header "Available Training Configurations"
    echo "======================================"
    echo ""
    echo "=== SmolLM3 Configurations ==="
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
    echo "=== GPT-OSS Configurations ==="
    echo "5. GPT-OSS Basic Training"
    echo "   - Model: openai/gpt-oss-20b"
    echo "   - Dataset: Multilingual-Thinking"
    echo "   - Epochs: 1"
    echo "   - Batch Size: 4"
    echo "   - Learning Rate: 2e-4"
    echo "   - LoRA + MXFP4 Quantization"
    echo "   - Optimized for multilingual reasoning"
    echo ""
    echo "6. GPT-OSS H100 Optimized"
    echo "   - Model: openai/gpt-oss-20b"
    echo "   - Dataset: Multilingual-Thinking"
    echo "   - Epochs: 2"
    echo "   - Batch Size: 8"
    echo "   - Learning Rate: 3e-4"
    echo "   - Enhanced LoRA (rank 16)"
    echo "   - Optimized for H100 performance"
    echo ""
    echo "7. GPT-OSS Multilingual Reasoning"
    echo "   - Model: openai/gpt-oss-20b"
    echo "   - Dataset: Multilingual-Thinking"
    echo "   - Epochs: 1"
    echo "   - Batch Size: 4"
    echo "   - Learning Rate: 2e-4"
    echo "   - Specialized for reasoning tasks"
    echo "   - Supports 10+ languages"
    echo ""
    echo "8. GPT-OSS Memory Optimized"
    echo "   - Model: openai/gpt-oss-20b"
    echo "   - Dataset: Multilingual-Thinking"
    echo "   - Epochs: 1"
    echo "   - Batch Size: 1 (effective 16 with accumulation)"
    echo "   - Learning Rate: 2e-4"
    echo "   - 4-bit quantization + reduced LoRA"
    echo "   - Optimized for limited GPU memory"
    echo ""
    echo "9. GPT-OSS OpenHermes-FR (Recommended)"
    echo "   - Model: openai/gpt-oss-20b"
    echo "   - Dataset: legmlai/openhermes-fr (800K French examples)"
    echo "   - Epochs: 1.5"
    echo "   - Batch Size: 6 (effective 36 with accumulation)"
    echo "   - Learning Rate: 2.5e-4"
    echo "   - Optimized for French language training"
    echo "   - Quality filtering enabled"
    echo ""
    echo "10. GPT-OSS OpenHermes-FR Memory Optimized"
    echo "   - Model: openai/gpt-oss-20b"
    echo "   - Dataset: legmlai/openhermes-fr (200K samples)"
    echo "   - Epochs: 1"
    echo "   - Batch Size: 2 (effective 32 with accumulation)"
    echo "   - Learning Rate: 2e-4"
    echo "   - Native MXFP4 quantization"
    echo "   - Memory optimized for 40-80GB GPUs"
    echo "   - Harmony format compatible"
    echo ""
    echo "10. GPT-OSS Custom Dataset"
    echo "   - Model: openai/gpt-oss-20b"
    echo "   - Dataset: User-defined (fully customizable)"
    echo "   - Epochs: Configurable"
    echo "   - Batch Size: Configurable"
    echo "   - Learning Rate: Configurable"
    echo "   - Maximum flexibility with all parameters"
    echo ""
    echo "11. Custom Configuration"
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
        "GPT-OSS Basic Training")
            MODEL_NAME="openai/gpt-oss-20b"
            DATASET_NAME="HuggingFaceH4/Multilingual-Thinking"
            MAX_EPOCHS=1
            BATCH_SIZE=4
            GRADIENT_ACCUMULATION_STEPS=4
            LEARNING_RATE=2e-4
            MAX_SEQ_LENGTH=2048
            CONFIG_FILE="config/train_gpt_oss_basic.py"
            ;;
        "GPT-OSS H100 Optimized")
            MODEL_NAME="openai/gpt-oss-20b"
            DATASET_NAME="HuggingFaceH4/Multilingual-Thinking"
            MAX_EPOCHS=2
            BATCH_SIZE=8
            GRADIENT_ACCUMULATION_STEPS=2
            LEARNING_RATE=3e-4
            MAX_SEQ_LENGTH=4096
            CONFIG_FILE="config/train_gpt_oss_h100_optimized.py"
            ;;
        "GPT-OSS Multilingual Reasoning")
            MODEL_NAME="openai/gpt-oss-20b"
            DATASET_NAME="HuggingFaceH4/Multilingual-Thinking"
            MAX_EPOCHS=1
            BATCH_SIZE=4
            GRADIENT_ACCUMULATION_STEPS=4
            LEARNING_RATE=2e-4
            MAX_SEQ_LENGTH=2048
            CONFIG_FILE="config/train_gpt_oss_multilingual_reasoning.py"
            ;;
        "GPT-OSS Memory Optimized")
            MODEL_NAME="openai/gpt-oss-20b"
            DATASET_NAME="HuggingFaceH4/Multilingual-Thinking"
            MAX_EPOCHS=1
            BATCH_SIZE=1
            GRADIENT_ACCUMULATION_STEPS=16
            LEARNING_RATE=2e-4
            MAX_SEQ_LENGTH=1024
            CONFIG_FILE="config/train_gpt_oss_memory_optimized.py"
            ;;
        "GPT-OSS OpenHermes-FR (Recommended)")
            MODEL_NAME="openai/gpt-oss-20b"
            DATASET_NAME="legmlai/openhermes-fr"
            MAX_EPOCHS=1.5
            BATCH_SIZE=6
            GRADIENT_ACCUMULATION_STEPS=6
            LEARNING_RATE=2.5e-4
            MAX_SEQ_LENGTH=3072
            CONFIG_FILE="config/train_gpt_oss_openhermes_fr.py"
            ;;
        "GPT-OSS OpenHermes-FR Memory Optimized")
            MODEL_NAME="openai/gpt-oss-20b"
            DATASET_NAME="legmlai/openhermes-fr"
            MAX_EPOCHS=1
            BATCH_SIZE=2
            GRADIENT_ACCUMULATION_STEPS=16
            LEARNING_RATE=2e-4
            MAX_SEQ_LENGTH=1024
            CONFIG_FILE="config/train_gpt_oss_openhermes_fr_memory_optimized.py"
            ;;
        "GPT-OSS Custom Dataset")
            MODEL_NAME="openai/gpt-oss-20b"
            DATASET_NAME="legmlai/openhermes-fr"  # Will be customizable
            MAX_EPOCHS=1
            BATCH_SIZE=4
            GRADIENT_ACCUMULATION_STEPS=4
            LEARNING_RATE=2e-4
            MAX_SEQ_LENGTH=2048
            CONFIG_FILE="config/train_gpt_oss_custom.py"
            get_custom_dataset_config
            ;;
        "Custom Configuration")
            get_custom_config
            ;;
    esac
}

# Function to get custom dataset configuration
get_custom_dataset_config() {
    print_step "GPT-OSS Custom Configuration"
    echo "======================================"
    
    echo "Configure your GPT-OSS training:"
    echo ""
    
    # Dataset Configuration
    print_info "📊 Dataset Configuration"
    get_input "Dataset name (HuggingFace format: username/dataset)" "legmlai/openhermes-fr" DATASET_NAME
    get_input "Dataset split" "train" DATASET_SPLIT
    
    echo ""
    echo "Dataset format options:"
    echo "1. OpenHermes-FR (prompt + accepted_completion fields)"
    echo "2. Messages format (chat conversations)"
    echo "3. Text format (plain text field)"
    echo "4. Custom format (specify field names)"
    echo ""
    
    select_option "Select dataset format:" "OpenHermes-FR" "Messages format" "Text format" "Custom format" DATASET_FORMAT
    
    case "$DATASET_FORMAT" in
        "OpenHermes-FR")
            INPUT_FIELD="prompt"
            TARGET_FIELD="accepted_completion"
            DATASET_FORMAT_CODE="openhermes_fr"
            FILTER_BAD_ENTRIES="true"
            ;;
        "Messages format")
            INPUT_FIELD="messages"
            TARGET_FIELD=""
            DATASET_FORMAT_CODE="messages"
            FILTER_BAD_ENTRIES="false"
            ;;
        "Text format")
            INPUT_FIELD="text"
            TARGET_FIELD=""
            DATASET_FORMAT_CODE="text"
            FILTER_BAD_ENTRIES="false"
            ;;
        "Custom format")
            get_input "Input field name" "prompt" INPUT_FIELD
            get_input "Target field name (leave empty if not needed)" "accepted_completion" TARGET_FIELD
            DATASET_FORMAT_CODE="custom"
            get_input "Filter bad entries? (true/false)" "false" FILTER_BAD_ENTRIES
            ;;
    esac
    
    # Dataset Filtering Options
    echo ""
    print_info "🔍 Dataset Filtering Options"
    get_input "Maximum samples to use (leave empty for all)" "" MAX_SAMPLES
    get_input "Minimum sequence length" "10" MIN_LENGTH
    get_input "Maximum sequence length (leave empty for auto)" "" MAX_LENGTH
    
    # Training Hyperparameters
    echo ""
    print_info "⚙️ Training Hyperparameters"
    get_input "Number of epochs" "1.0" NUM_EPOCHS
    get_input "Batch size per device" "4" BATCH_SIZE
    get_input "Gradient accumulation steps" "4" GRAD_ACCUM_STEPS
    get_input "Learning rate" "2e-4" LEARNING_RATE
    get_input "Minimum learning rate" "2e-5" MIN_LR
    get_input "Weight decay" "0.01" WEIGHT_DECAY
    get_input "Warmup ratio" "0.03" WARMUP_RATIO
    
    # Sequence Length
    echo ""
    print_info "📏 Sequence Configuration"
    get_input "Maximum sequence length" "2048" MAX_SEQ_LENGTH
    
    # LoRA Configuration
    echo ""
    print_info "🎛️ LoRA Configuration"
    get_input "LoRA rank" "16" LORA_RANK
    get_input "LoRA alpha" "32" LORA_ALPHA
    get_input "LoRA dropout" "0.05" LORA_DROPOUT
    
    # Memory & Performance
    echo ""
    print_info "💾 Memory & Performance"
    select_option "Mixed precision:" "BF16 (recommended)" "FP16" "FP32" MIXED_PRECISION
    get_input "Data loading workers" "4" NUM_WORKERS
    select_option "Quantization:" "MXFP4 (default)" "4-bit BNB" "None" QUANTIZATION_TYPE
    
    # Advanced Options
    echo ""
    echo "Advanced options (press Enter for defaults):"
    get_input "Max gradient norm" "1.0" MAX_GRAD_NORM
    get_input "Logging steps" "10" LOGGING_STEPS
    get_input "Evaluation steps" "100" EVAL_STEPS
    get_input "Save steps" "500" SAVE_STEPS
    
    # Update the custom config file with user's choices
    update_enhanced_gpt_oss_config
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

# Function to update enhanced GPT-OSS config with user choices
update_enhanced_gpt_oss_config() {
    print_info "Generating enhanced custom GPT-OSS configuration..."
    
    # Process mixed precision setting
    case "$MIXED_PRECISION" in
        "BF16 (recommended)")
            FP16="False"
            BF16="True"
            ;;
        "FP16")
            FP16="True"
            BF16="False"
            ;;
        "FP32")
            FP16="False"
            BF16="False"
            ;;
    esac
    
    # Process quantization setting
    case "$QUANTIZATION_TYPE" in
        "MXFP4 (default)")
            USE_QUANTIZATION="True"
            QUANTIZATION_CONFIG='{"dequantize": True, "load_in_4bit": False}'
            ;;
        "4-bit BNB")
            USE_QUANTIZATION="True"
            QUANTIZATION_CONFIG='{"dequantize": False, "load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True, "bnb_4bit_quant_type": "nf4"}'
            ;;
        "None")
            USE_QUANTIZATION="False"
            QUANTIZATION_CONFIG='{"dequantize": False, "load_in_4bit": False}'
            ;;
    esac
    
    # Create enhanced config file with all user choices
    cat > "$CONFIG_FILE" << EOF
"""
GPT-OSS Enhanced Custom Training Configuration - Generated by launch.sh
Dataset: $DATASET_NAME ($DATASET_FORMAT)
Optimized for: ${DATASET_FORMAT} format with full customization
"""

from config.train_gpt_oss_custom import GPTOSSEnhancedCustomConfig

# Create enhanced config with all customizations
config = GPTOSSEnhancedCustomConfig(
    # ============================================================================
    # DATASET CONFIGURATION
    # ============================================================================
    dataset_name="$DATASET_NAME",
    dataset_split="$DATASET_SPLIT",
    dataset_format="$DATASET_FORMAT_CODE",
    input_field="$INPUT_FIELD",
    target_field=$(if [ -n "$TARGET_FIELD" ]; then echo "\"$TARGET_FIELD\""; else echo "None"; fi),
    filter_bad_entries=$FILTER_BAD_ENTRIES,
    max_samples=$(if [ -n "$MAX_SAMPLES" ]; then echo "$MAX_SAMPLES"; else echo "None"; fi),
    min_length=$MIN_LENGTH,
    max_length=$(if [ -n "$MAX_LENGTH" ]; then echo "$MAX_LENGTH"; else echo "None"; fi),
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    num_train_epochs=$NUM_EPOCHS,
    batch_size=$BATCH_SIZE,
    gradient_accumulation_steps=$GRAD_ACCUM_STEPS,
    learning_rate=$LEARNING_RATE,
    min_lr=$MIN_LR,
    weight_decay=$WEIGHT_DECAY,
    warmup_ratio=$WARMUP_RATIO,
    max_grad_norm=$MAX_GRAD_NORM,
    
    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    max_seq_length=$MAX_SEQ_LENGTH,
    
    # ============================================================================
    # MIXED PRECISION
    # ============================================================================
    fp16=$FP16,
    bf16=$BF16,
    
    # ============================================================================
    # LORA CONFIGURATION
    # ============================================================================
    lora_config={
        "r": $LORA_RANK,
        "lora_alpha": $LORA_ALPHA,
        "lora_dropout": $LORA_DROPOUT,
        "target_modules": "all-linear",
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    
    # ============================================================================
    # QUANTIZATION CONFIGURATION
    # ============================================================================
    use_quantization=$USE_QUANTIZATION,
    quantization_config=$QUANTIZATION_CONFIG,
    
    # ============================================================================
    # PERFORMANCE CONFIGURATION
    # ============================================================================
    dataloader_num_workers=$NUM_WORKERS,
    dataloader_pin_memory=True,
    group_by_length=True,
    
    # ============================================================================
    # LOGGING & EVALUATION
    # ============================================================================
    logging_steps=$LOGGING_STEPS,
    eval_steps=$EVAL_STEPS,
    save_steps=$SAVE_STEPS,
    
    # ============================================================================
    # RUNTIME CONFIGURATION
    # ============================================================================
    experiment_name="$EXPERIMENT_NAME",
    trackio_url="$TRACKIO_URL",
    dataset_repo="$TRACKIO_DATASET_REPO",
    enable_tracking=True,
)
EOF
    
    print_status "Enhanced GPT-OSS configuration generated successfully!"
    print_info "Configuration saved to: $CONFIG_FILE"
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
    # Trainer type selection
    trainer_type="$TRAINER_TYPE",
    
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

# Step 1: Get user credentials (write and read tokens)
print_step "Step 1: User Authentication"
echo "================================"

print_info "You'll need two Hugging Face tokens:"
echo "1. Write Token - Used initially for training and creating repositories"
echo "2. Read Token - Will replace the write token in Trackio Space after training for security"
echo ""
print_info "The pipeline will start with the write token in HF_TOKEN, then switch to read token automatically."
echo ""

print_info "Getting Write Token (for training operations)..."
get_secure_token_input "Enter your Hugging Face WRITE token (get from https://huggingface.co/settings/tokens)" HF_WRITE_TOKEN "Write"

print_info "Getting Read Token (for Trackio Space security)..."
get_secure_token_input "Enter your Hugging Face READ token (get from https://huggingface.co/settings/tokens)" HF_READ_TOKEN "Read"

# Validate write token and get username automatically
print_info "Validating write token and getting username..."
if validate_hf_token_and_get_username "$HF_WRITE_TOKEN"; then
    print_status "Write token validated successfully"
    print_info "Username: $HF_USERNAME"
else
    print_error "Invalid write token. Please check your token and try again."
    exit 1
fi

# Validate read token belongs to same user
print_info "Validating read token..."
if validate_hf_token_and_get_username "$HF_READ_TOKEN"; then
    READ_USERNAME="$HF_USERNAME"
    if [ "$READ_USERNAME" = "$HF_USERNAME" ]; then
        print_status "Read token validated successfully"
        print_info "Both tokens belong to user: $HF_USERNAME"
    else
        print_error "Token mismatch: write token user ($HF_USERNAME) != read token user ($READ_USERNAME)"
        print_error "Both tokens must belong to the same user"
        exit 1
    fi
else
    print_error "Invalid read token. Please check your token and try again."
    exit 1
fi

# Set the main HF_TOKEN to write token for training operations (will be switched later)
HF_TOKEN="$HF_WRITE_TOKEN"

# Step 2: Select training configuration
print_step "Step 2: Training Configuration"
echo "=================================="

show_training_configs
select_option "Select training configuration:" "Basic Training" "H100 Lightweight (Rapid)" "A100 Large Scale" "Multiple Passes" "GPT-OSS Basic Training" "GPT-OSS H100 Optimized" "GPT-OSS Multilingual Reasoning" "GPT-OSS Memory Optimized" "GPT-OSS OpenHermes-FR (Recommended)" "GPT-OSS OpenHermes-FR Memory Optimized" "GPT-OSS Custom Dataset" "Custom Configuration" TRAINING_CONFIG_TYPE

get_training_config "$TRAINING_CONFIG_TYPE"

# Step 3: Get experiment details
print_step "Step 3: Experiment Details"
echo "=============================="

get_input "Experiment name" "smollm3_finetune_$(date +%Y%m%d_%H%M%S)" EXPERIMENT_NAME

# Configure model repository name (customizable)
print_info "Setting up model repository name..."
DEFAULT_REPO_NAME="$HF_USERNAME/smolfactory-$(date +%Y%m%d)"
get_input "Model repository name (Hugging Face format: username/repo)" "$DEFAULT_REPO_NAME" REPO_NAME
print_status "Model repository: $REPO_NAME"

# Automatically create dataset repository
print_info "Setting up Trackio dataset repository automatically..."

# Set default dataset repository
TRACKIO_DATASET_REPO="$HF_USERNAME/trackio-experiments"

# Ask if user wants to customize dataset name
echo ""
echo "Dataset repository options:"
echo "1. Use default name (trackio-experiments)"
echo "2. Customize dataset name"
echo ""
read -p "Choose option (1/2): " dataset_option

if [ "$dataset_option" = "2" ]; then
    get_input "Custom dataset name (without username)" "trackio-experiments" CUSTOM_DATASET_NAME
    if python3 scripts/dataset_tonic/setup_hf_dataset.py "$HF_TOKEN" "$CUSTOM_DATASET_NAME" 2>/dev/null; then
        # Update with the actual repository name from the script
        TRACKIO_DATASET_REPO="$TRACKIO_DATASET_REPO"
        print_status "Custom dataset repository created successfully"
    else
        print_warning "Custom dataset creation failed, using default"
        if python3 scripts/dataset_tonic/setup_hf_dataset.py "$HF_TOKEN" 2>/dev/null; then
            TRACKIO_DATASET_REPO="$TRACKIO_DATASET_REPO"
            print_status "Default dataset repository created successfully"
        else
            print_warning "Automatic dataset creation failed, using default"
            TRACKIO_DATASET_REPO="$HF_USERNAME/trackio-experiments"
        fi
    fi
else
    if python3 scripts/dataset_tonic/setup_hf_dataset.py "$HF_TOKEN" 2>/dev/null; then
        TRACKIO_DATASET_REPO="$TRACKIO_DATASET_REPO"
        print_status "Dataset repository created successfully"
    else
        print_warning "Automatic dataset creation failed, using default"
        TRACKIO_DATASET_REPO="$HF_USERNAME/trackio-experiments"
    fi
fi

# Ensure TRACKIO_DATASET_REPO is always set
if [ -z "$TRACKIO_DATASET_REPO" ]; then
    TRACKIO_DATASET_REPO="$HF_USERNAME/trackio-experiments"
    print_warning "Dataset repository not set, using default: $TRACKIO_DATASET_REPO"
fi

# Step 3.5: Select trainer type
print_step "Step 3.5: Trainer Type Selection"
echo "===================================="

echo "Select the type of training to perform:"
echo "1. SFT (Supervised Fine-tuning) - Standard instruction tuning"
echo "   - Uses SFTTrainer for instruction following"
echo "   - Suitable for most fine-tuning tasks"
echo "   - Optimized for instruction datasets"
echo ""
echo "2. DPO (Direct Preference Optimization) - Preference-based training"
echo "   - Uses DPOTrainer for preference learning"
echo "   - Requires preference datasets (chosen/rejected pairs)"
echo "   - Optimizes for human preferences"
echo ""

select_option "Select trainer type:" "SFT" "DPO" TRAINER_TYPE

# Convert trainer type to lowercase for the training script
TRAINER_TYPE_LOWER=$(echo "$TRAINER_TYPE" | tr '[:upper:]' '[:lower:]')

# Step 4: Training parameters
print_step "Step 4: Training Parameters"
echo "==============================="

echo "Current configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME"
echo "  Trainer Type: $TRAINER_TYPE"
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
echo "  User: $HF_USERNAME (auto-detected from token)"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME"
echo "  Training Config: $TRAINING_CONFIG_TYPE"
echo "  Trainer Type: $TRAINER_TYPE"
if [ "$TRAINING_CONFIG_TYPE" = "H100 Lightweight (Rapid)" ]; then
    echo "  Dataset Sample Size: ${DATASET_SAMPLE_SIZE:-80000}"
fi
echo "  Epochs: $MAX_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Model Repo: $REPO_NAME (auto-generated)"
echo "  Author: $AUTHOR_NAME"
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

# Check if we're already root or if sudo is available
if [ "$EUID" -eq 0 ]; then
    # Already root, no need for sudo
    print_info "Running as root, skipping sudo..."
    apt-get update
    apt-get install -y git curl wget unzip python3-pip python3-venv
elif command -v sudo >/dev/null 2>&1; then
    # sudo is available, use it
    print_info "Using sudo for system dependencies..."
    sudo apt-get update
    sudo apt-get install -y git curl wget unzip python3-pip python3-venv
else
    # No sudo available, try without it
    print_warning "sudo not available, attempting to install without sudo..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y git curl wget unzip python3-pip python3-venv
    else
        print_warning "apt-get not available, skipping system dependencies..."
        print_info "Please ensure git, curl, wget, unzip, python3-pip, and python3-venv are installed"
    fi
fi

# Set environment variables before creating virtual environment
print_info "Setting up environment variables..."
export HF_TOKEN="$HF_TOKEN"
export TRACKIO_DATASET_REPO="$TRACKIO_DATASET_REPO"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_USERNAME="$HF_USERNAME"

print_info "Creating Python virtual environment..."
python3 -m venv smollm3_env
source smollm3_env/bin/activate

# Re-export environment variables in the virtual environment
print_info "Configuring environment variables in virtual environment..."
export HF_TOKEN="$HF_TOKEN"
export TRACKIO_DATASET_REPO="$TRACKIO_DATASET_REPO"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_USERNAME="$HF_USERNAME"

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

print_info "Setting up Hugging Face token for Python API..."
print_status "HF token configured for Python API usage"
print_info "Username: $HF_USERNAME (auto-detected from token)"
print_info "Token available in environment: ${HF_TOKEN:0:10}...${HF_TOKEN: -4}"

# Verify token is available in the virtual environment
print_info "Verifying token availability in virtual environment..."
if [ -n "$HF_TOKEN" ] && [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    print_status "✅ Token properly configured in virtual environment"
    print_info "  HF_TOKEN: ${HF_TOKEN:0:10}...${HF_TOKEN: -4} (currently using WRITE token)"
    print_info "  HUGGING_FACE_HUB_TOKEN: ${HUGGING_FACE_HUB_TOKEN:0:10}...${HUGGING_FACE_HUB_TOKEN: -4}"
    print_info "  Will be switched to READ token after training for security"
else
    print_error "❌ Token not properly configured in virtual environment"
    print_error "Please check your token and try again"
    exit 1
fi

# Configure git for HF operations
print_step "Step 8.1: Git Configuration"
echo "================================"

print_info "Configuring git for Hugging Face operations..."

# Get user's email for git configuration
get_input "Enter the email you used to register your account at huggingface for git configuration" "" GIT_EMAIL

# Configure git locally (not globally) for this project
git config user.email "$GIT_EMAIL"
git config user.name "$HF_USERNAME"

# Verify git configuration
print_info "Verifying git configuration..."
if git config user.email && git config user.name; then
    print_status "Git configured successfully"
    print_info "  Email: $(git config user.email)"
    print_info "  Name: $(git config user.name)"
else
    print_error "Failed to configure git"
    exit 1
fi

# Step 8.2: Author Information for Model Card
print_step "Step 8.2: Author Information"
echo "================================="

print_info "This information will be used in the model card and citation."
get_input "Author name for model card" "$HF_USERNAME" AUTHOR_NAME

print_info "Model description will be used in the model card and repository."
get_input "Model description" "A fine-tuned version of SmolLM3-3B for improved french language text generation and conversation capabilities." MODEL_DESCRIPTION

# Step 9: Deploy Trackio Space (automated)
print_step "Step 9: Deploying Trackio Space"
echo "==================================="

cd scripts/trackio_tonic

print_info "Deploying Trackio Space ..."
print_info "Space name: $TRACKIO_SPACE_NAME"
print_info "Username will be auto-detected from token"
print_info "Secrets will be set automatically via API"

# Ensure environment variables are available for the script
export HF_TOKEN="$HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_USERNAME="$HF_USERNAME"

# Run deployment script with automated features
python deploy_trackio_space.py "$TRACKIO_SPACE_NAME" "$HF_TOKEN" "$GIT_EMAIL" "$HF_USERNAME" "$TRACKIO_DATASET_REPO"

print_status "Trackio Space deployed: $TRACKIO_URL"

# Step 10: Setup HF Dataset (automated)
print_step "Step 10: Setting up HF Dataset"
echo "=================================="

cd ../dataset_tonic
print_info "Setting up HF Dataset with automated features..."
print_info "Username will be auto-detected from token"
print_info "Dataset repository: $TRACKIO_DATASET_REPO"

# Ensure environment variables are available for the script
export HF_TOKEN="$HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_USERNAME="$HF_USERNAME"

python setup_hf_dataset.py "$HF_TOKEN"

# Step 11: Configure Trackio (automated)
print_step "Step 11: Configuring Trackio"
echo "================================="

cd ../trackio_tonic
print_info "Configuring Trackio ..."
print_info "Username will be auto-detected from token"

# Ensure environment variables are available for the script
export HF_TOKEN="$HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_USERNAME="$HF_USERNAME"

python configure_trackio.py

# Step 12: Training Configuration
print_step "Step 12: Training Configuration"
echo "==================================="

cd ../..
print_info "Using existing configuration file: $CONFIG_FILE"

# Step 13: Dataset Configuration
print_step "Step 13: Dataset Configuration"
echo "=================================="

print_info "Dataset will be loaded directly by src/data.py during training"
print_info "Dataset: $DATASET_NAME"
if [ "$TRAINING_CONFIG_TYPE" = "H100 Lightweight (Rapid)" ]; then
    print_info "Sample size: ${DATASET_SAMPLE_SIZE:-80000} (will be handled by data.py)"
fi

# Step 14: Training Parameters
print_step "Step 14: Training Parameters"
echo "================================"

print_info "Training parameters will be loaded from configuration file"
print_info "Model: $MODEL_NAME"
print_info "Dataset: $DATASET_NAME"
print_info "Batch size: $BATCH_SIZE"
print_info "Learning rate: $LEARNING_RATE"

# Step 14.5: Define Output Directory
print_step "Step 14.5: Output Directory Configuration"
echo "============================================="

# Define the output directory for training results
OUTPUT_DIR="./outputs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
print_info "Training output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_status "Output directory created: $OUTPUT_DIR"

# Step 15: Start training
print_step "Step 15: Starting Training"
echo "=============================="

print_info "Starting training with configuration: $CONFIG_FILE"
print_info "Experiment: $EXPERIMENT_NAME"
print_info "Output: $OUTPUT_DIR"
print_info "Trackio: $TRACKIO_URL"

# Ensure environment variables are available for training
export HF_WRITE_TOKEN="$HF_WRITE_TOKEN"
export HF_READ_TOKEN="$HF_READ_TOKEN"
export HF_TOKEN="$HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_USERNAME="$HF_USERNAME"
export TRACKIO_DATASET_REPO="$TRACKIO_DATASET_REPO"
export OUTPUT_DIR="$OUTPUT_DIR"

# Run the appropriate training script based on model type
if [[ "$MODEL_NAME" == *"gpt-oss"* ]]; then
    print_info "Using GPT-OSS specialized training script..."
    python scripts/training/train_gpt_oss.py \
        --config "$CONFIG_FILE" \
        --experiment-name "$EXPERIMENT_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --trackio-url "$TRACKIO_URL" \
        --trainer-type "$TRAINER_TYPE_LOWER"
else
    print_info "Using standard SmolLM3 training script..."
    python scripts/training/train.py \
        --config "$CONFIG_FILE" \
        --experiment-name "$EXPERIMENT_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --trackio-url "$TRACKIO_URL" \
        --trainer-type "$TRAINER_TYPE_LOWER"
fi

# Step 16: Push model to Hugging Face Hub
print_step "Step 16: Pushing Model to HF Hub"
echo "====================================="

print_info "Pushing model to: $REPO_NAME"
print_info "Checkpoint: $OUTPUT_DIR"

# Ensure environment variables are available for model push
export HF_WRITE_TOKEN="$HF_WRITE_TOKEN"
export HF_READ_TOKEN="$HF_READ_TOKEN"
export HF_TOKEN="$HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_USERNAME="$HF_USERNAME"
export TRACKIO_DATASET_REPO="$TRACKIO_DATASET_REPO"
export OUTPUT_DIR="$OUTPUT_DIR"

# Run the appropriate push script based on model type
if [[ "$MODEL_NAME" == *"gpt-oss"* ]]; then
    print_info "Using GPT-OSS specialized push script..."
    python scripts/model_tonic/push_gpt_oss_to_huggingface.py "$OUTPUT_DIR" "$REPO_NAME" \
        --token "$HF_TOKEN" \
        --trackio-url "$TRACKIO_URL" \
        --experiment-name "$EXPERIMENT_NAME" \
        --dataset-repo "$TRACKIO_DATASET_REPO" \
        --author-name "$AUTHOR_NAME" \
        --model-description "$MODEL_DESCRIPTION" \
        --training-config-type "$TRAINING_CONFIG_TYPE" \
        --model-name "$MODEL_NAME" \
        --dataset-name "$DATASET_NAME" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --max-epochs "$MAX_EPOCHS" \
        --max-seq-length "$MAX_SEQ_LENGTH" \
        --trainer-type "$TRAINER_TYPE"
else
    print_info "Using standard SmolLM3 push script..."
    python scripts/model_tonic/push_to_huggingface.py "$OUTPUT_DIR" "$REPO_NAME" \
        --token "$HF_TOKEN" \
        --trackio-url "$TRACKIO_URL" \
        --experiment-name "$EXPERIMENT_NAME" \
        --dataset-repo "$TRACKIO_DATASET_REPO" \
        --author-name "$AUTHOR_NAME" \
        --model-description "$MODEL_DESCRIPTION" \
        --training-config-type "$TRAINING_CONFIG_TYPE" \
        --model-name "$MODEL_NAME" \
        --dataset-name "$DATASET_NAME" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --max-epochs "$MAX_EPOCHS" \
        --max-seq-length "$MAX_SEQ_LENGTH" \
        --trainer-type "$TRAINER_TYPE"
fi

# Step 16.5: Switch Trackio Space to Read Token (Security)
print_step "Step 16.5: Switching to Read Token for Security"
echo "===================================================="

print_info "Switching Trackio Space HF_TOKEN from write token to read token for security..."
print_info "This ensures the space can only read datasets, not write to repositories"

# Ensure environment variables are available for token switch
export HF_TOKEN="$HF_WRITE_TOKEN"  # Use write token to update space
export HUGGING_FACE_HUB_TOKEN="$HF_WRITE_TOKEN"
export HF_USERNAME="$HF_USERNAME"

# Switch HF_TOKEN in Trackio Space from write to read token
cd scripts/trackio_tonic
python switch_to_read_token.py "$HF_USERNAME/$TRACKIO_SPACE_NAME" "$HF_READ_TOKEN" "$HF_WRITE_TOKEN"

if [ $? -eq 0 ]; then
    print_status "✅ Successfully switched Trackio Space HF_TOKEN to read token"
    print_info "🔒 Space now uses read-only permissions for security"
else
    print_warning "⚠️ Failed to switch to read token, but continuing with pipeline"
    print_info "You can manually switch the token in your Space settings later"
fi

cd ../..

# Step 17: Deploy Demo Space
print_step "Step 17: Deploying Demo Space"
echo "=================================="

# Ask user if they want to deploy a demo space
get_input "Do you want to deploy a demo space to test your model? (y/n)" "y" "DEPLOY_DEMO"

if [ "$DEPLOY_DEMO" = "y" ] || [ "$DEPLOY_DEMO" = "Y" ]; then
    print_info "Deploying demo space for model testing..."
    
    # Use main model for demo (no quantization)
    DEMO_MODEL_ID="$REPO_NAME"
    DEMO_SUBFOLDER=""
    
    # Ensure environment variables are available for demo deployment
export HF_TOKEN="$HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_USERNAME="$HF_USERNAME"
    
    print_info "Deploying demo space for model: $DEMO_MODEL_ID"
    print_info "Using subfolder: $DEMO_SUBFOLDER"
    
    python scripts/deploy_demo_space.py \
        --hf-token "$HF_TOKEN" \
        --hf-username "$HF_USERNAME" \
        --model-id "$DEMO_MODEL_ID" \
        --subfolder "$DEMO_SUBFOLDER" \
        --space-name "${REPO_NAME}-demo"
    
    if [ $? -eq 0 ]; then
        DEMO_SPACE_URL="https://huggingface.co/spaces/$HF_USERNAME/${REPO_NAME}-demo"
        print_status "✅ Demo space deployed successfully: $DEMO_SPACE_URL"
    else
        print_warning "⚠️ Demo space deployment failed, but continuing with pipeline"
    fi
else
    print_info "Skipping demo space deployment"
fi

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
- **Trainer Type**: $TRAINER_TYPE
- **Security**: Single HF_TOKEN switched from write to read token
$(if [ "$TRAINING_CONFIG_TYPE" = "H100 Lightweight (Rapid)" ]; then
echo "- **Dataset Sample Size**: ${DATASET_SAMPLE_SIZE:-80000}"
fi)

## Training Parameters
- **Batch Size**: $BATCH_SIZE
- **Gradient Accumulation**: $GRADIENT_ACCUMULATION_STEPS
- **Learning Rate**: $LEARNING_RATE
- **Max Epochs**: $MAX_EPOCHS
- **Sequence Length**: $MAX_SEQ_LENGTH

## Results
- **Model Repository**: https://huggingface.co/$REPO_NAME
- **Trackio Monitoring**: $TRACKIO_URL
- **Experiment Data**: https://huggingface.co/datasets/$TRACKIO_DATASET_REPO
- **Security**: Trackio Space HF_TOKEN switched to read-only token for security
$(if [ "$DEPLOY_DEMO" = "y" ] || [ "$DEPLOY_DEMO" = "Y" ]; then
echo "- **Demo Space**: https://huggingface.co/spaces/$HF_USERNAME/${REPO_NAME}-demo"
fi)

## Next Steps
1. Monitor training progress in your Trackio Space
2. Check the model repository on Hugging Face Hub
3. Use the model in your applications
4. Share your results with the community

## Files Created
- Training configuration: \`$CONFIG_FILE\`
- Model checkpoint: \`$OUTPUT_DIR/\`
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
$(if [ "$DEPLOY_DEMO" = "y" ] || [ "$DEPLOY_DEMO" = "Y" ]; then
echo "🎮 Demo: https://huggingface.co/spaces/$HF_USERNAME/${REPO_NAME}-demo"
fi)
echo ""
echo "📋 Summary report saved to: training_summary.md"
echo ""
echo "🚀 Next steps:"
echo "1. Monitor training progress in your Trackio Space"
echo "2. Check the model repository on Hugging Face Hub"
echo "3. Your Trackio Space HF_TOKEN is now secured with read-only permissions"
$(if [ "$DEPLOY_DEMO" = "y" ] || [ "$DEPLOY_DEMO" = "Y" ]; then
echo "3. Make your huggingface space a ZeroGPU Space & Test your model"
fi)
echo "5. Use the model in your applications"
echo "6. Share your results with the community"
echo ""
print_status "Pipeline completed successfully!" 