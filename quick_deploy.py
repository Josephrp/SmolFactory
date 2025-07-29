#!/usr/bin/env python3
"""
Quick Model Deployment Script
Direct deployment without argument parsing issues
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Direct deployment without argument parsing"""
    
    # Configuration
    MODEL_PATH = "/output-checkpoint"
    REPO_NAME = "Tonic/smollm3-finetuned"
    HF_TOKEN = os.getenv('HF_TOKEN')
    
    if not HF_TOKEN:
        logger.error("❌ HF_TOKEN not set")
        return 1
    
    if not Path(MODEL_PATH).exists():
        logger.error(f"❌ Model path not found: {MODEL_PATH}")
        return 1
    
    logger.info("✅ Model files validated")
    
    # Import and run the recovery pipeline directly
    try:
        from recover_model import ModelRecoveryPipeline
        
        # Initialize pipeline
        pipeline = ModelRecoveryPipeline(
            model_path=MODEL_PATH,
            repo_name=REPO_NAME,
            hf_token=HF_TOKEN,
            private=False,
            quantize=True,
            quant_types=["int8_weight_only", "int4_weight_only"],
            author_name="Tonic",
            model_description="A fine-tuned SmolLM3 model for improved text generation and conversation capabilities"
        )
        
        # Run the complete pipeline
        success = pipeline.run_complete_pipeline()
        
        if success:
            logger.info("✅ Model deployment completed successfully!")
            logger.info(f"🌐 View your model at: https://huggingface.co/{REPO_NAME}")
            return 0
        else:
            logger.error("❌ Model deployment failed!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error during deployment: {e}")
        return 1

if __name__ == "__main__":
    exit(main())