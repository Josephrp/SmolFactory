#!/bin/bash
# Cloud Model Recovery and Deployment Script
# Run this on your cloud instance to recover and deploy your trained model

set -e  # Exit on any error

echo "🚀 Starting cloud model recovery and deployment..."

# Configuration
MODEL_PATH="/output-checkpoint"
REPO_NAME="your-username/smollm3-finetuned"  # Change this to your HF username and desired repo name
HF_TOKEN="${HF_TOKEN}"  # Set this environment variable
PRIVATE=false  # Set to true if you want a private repository

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "$MODEL_PATH" ]; then
    print_error "Model path not found: $MODEL_PATH"
    exit 1
fi

print_status "Found model at: $MODEL_PATH"

# Check for required files
print_status "Validating model files..."
if [ ! -f "$MODEL_PATH/config.json" ]; then
    print_error "config.json not found"
    exit 1
fi

if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
    print_error "model.safetensors.index.json not found"
    exit 1
fi

if [ ! -f "$MODEL_PATH/tokenizer.json" ]; then
    print_error "tokenizer.json not found"
    exit 1
fi

print_success "Model files validated"

# Check HF token
if [ -z "$HF_TOKEN" ]; then
    print_error "HF_TOKEN environment variable not set"
    print_status "Please set your Hugging Face token:"
    print_status "export HF_TOKEN=your_token_here"
    exit 1
fi

print_success "HF Token found"

# Install required packages if not already installed
print_status "Checking dependencies..."
python3 -c "import torchao" 2>/dev/null || {
    print_status "Installing torchao..."
    pip install torchao
}

python3 -c "import huggingface_hub" 2>/dev/null || {
    print_status "Installing huggingface_hub..."
    pip install huggingface_hub
}

print_success "Dependencies checked"

# Run the recovery script
print_status "Running model recovery and deployment pipeline..."

python3 recover_model.py \
    "$MODEL_PATH" \
    "$REPO_NAME" \
    --hf-token "$HF_TOKEN" \
    --private "$PRIVATE" \
    --quant-types int8_weight_only int4_weight_only \
    --author-name "Your Name" \
    --model-description "A fine-tuned SmolLM3 model for improved text generation and conversation capabilities"

if [ $? -eq 0 ]; then
    print_success "Model recovery and deployment completed successfully!"
    print_success "View your model at: https://huggingface.co/$REPO_NAME"
    print_success "Quantized models available at:"
    print_success "  - https://huggingface.co/$REPO_NAME/int8 (GPU optimized)"
    print_success "  - https://huggingface.co/$REPO_NAME/int4 (CPU optimized)"
else
    print_error "Model recovery and deployment failed!"
    exit 1
fi

print_success "🎉 All done! Your model has been successfully recovered and deployed to Hugging Face Hub."