#!/usr/bin/env python3
"""
Quantize and Push Script
Quantizes the uploaded model and pushes quantized versions to the same repository
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
    """Quantize and push the model"""
    
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
    
    # Import and run quantization
    try:
        from scripts.model_tonic.quantize_model import ModelQuantizer
        
        # Quantization types to process
        quant_types = ["int8_weight_only", "int4_weight_only"]
        
        success_count = 0
        total_count = len(quant_types)
        
        for quant_type in quant_types:
            logger.info(f"🔄 Processing quantization type: {quant_type}")
            
            # Initialize quantizer
            quantizer = ModelQuantizer(
                model_path=MODEL_PATH,
                repo_name=REPO_NAME,
                token=HF_TOKEN,
                private=False,
                hf_token=HF_TOKEN
            )
            
            # Perform quantization and push
            success = quantizer.quantize_and_push(
                quant_type=quant_type,
                device="auto",
                group_size=128
            )
            
            if success:
                logger.info(f"✅ {quant_type} quantization and push completed")
                success_count += 1
            else:
                logger.error(f"❌ {quant_type} quantization and push failed")
        
        logger.info(f"📊 Quantization summary: {success_count}/{total_count} successful")
        
        if success_count > 0:
            logger.info("✅ Quantization completed successfully!")
            logger.info(f"🌐 View your models at: https://huggingface.co/{REPO_NAME}")
            logger.info("📊 Quantized models available at:")
            logger.info(f"  - https://huggingface.co/{REPO_NAME}/int8 (GPU optimized)")
            logger.info(f"  - https://huggingface.co/{REPO_NAME}/int4 (CPU optimized)")
            return 0
        else:
            logger.error("❌ All quantization attempts failed!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error during quantization: {e}")
        return 1

if __name__ == "__main__":
    exit(main())