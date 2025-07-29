#!/usr/bin/env python3
"""
Cloud Model Deployment Script
Run this directly on your cloud instance to deploy your trained model
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main deployment function"""
    
    # Configuration - CHANGE THESE VALUES
    MODEL_PATH = "/output-checkpoint"
    REPO_NAME = "your-username/smollm3-finetuned"  # Change to your HF username and desired repo name
    HF_TOKEN = os.getenv('HF_TOKEN')
    PRIVATE = False  # Set to True for private repository
    
    # Validate configuration
    if not HF_TOKEN:
        logger.error("❌ HF_TOKEN environment variable not set")
        logger.info("Please set your Hugging Face token:")
        logger.info("export HF_TOKEN=your_token_here")
        return 1
    
    if not Path(MODEL_PATH).exists():
        logger.error(f"❌ Model path not found: {MODEL_PATH}")
        return 1
    
    # Check for required files
    required_files = ['config.json', 'model.safetensors.index.json', 'tokenizer.json']
    for file in required_files:
        if not (Path(MODEL_PATH) / file).exists():
            logger.error(f"❌ Required file not found: {file}")
            return 1
    
    logger.info("✅ Model files validated")
    
    # Install dependencies if needed
    try:
        import torchao
        logger.info("✅ torchao available")
    except ImportError:
        logger.info("📦 Installing torchao...")
        os.system("pip install torchao")
    
    try:
        import huggingface_hub
        logger.info("✅ huggingface_hub available")
    except ImportError:
        logger.info("📦 Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
    
    # Run the recovery script
    logger.info("🚀 Starting model deployment...")
    
    cmd = [
        sys.executable, "recover_model.py",
        MODEL_PATH,
        REPO_NAME,
        "--hf-token", HF_TOKEN,
        "--quant-types", "int8_weight_only", "int4_weight_only",
        "--author-name", "Your Name",
        "--model-description", "A fine-tuned SmolLM3 model for improved text generation and conversation capabilities"
    ]
    
    if PRIVATE:
        cmd.append("--private")
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Run the command
    result = os.system(' '.join(cmd))
    
    if result == 0:
        logger.info("✅ Model deployment completed successfully!")
        logger.info(f"🌐 View your model at: https://huggingface.co/{REPO_NAME}")
        logger.info("📊 Quantized models available at:")
        logger.info(f"  - https://huggingface.co/{REPO_NAME}/int8 (GPU optimized)")
        logger.info(f"  - https://huggingface.co/{REPO_NAME}/int4 (CPU optimized)")
        return 0
    else:
        logger.error("❌ Model deployment failed!")
        return 1

if __name__ == "__main__":
    exit(main())