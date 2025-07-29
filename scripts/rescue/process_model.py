#!/usr/bin/env python3
"""
Model Processing Script
Processes recovered model with quantization and pushing to HF Hub
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelProcessor:
    """Process recovered model with quantization and pushing"""
    
    def __init__(self, model_path: str = "recovered_model"):
        self.model_path = Path(model_path)
        self.hf_token = os.getenv('HF_TOKEN')
        
    def validate_model(self) -> bool:
        """Validate that the model can be loaded"""
        try:
            logger.info("🔍 Validating model loading...")
            
            # Try to load the model
            cmd = [
                sys.executable, "-c",
                "from transformers import AutoModelForCausalLM; "
                "model = AutoModelForCausalLM.from_pretrained('recovered_model', "
                "torch_dtype='auto', device_map='auto'); "
                "print('✅ Model loaded successfully')"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Model validation successful")
                return True
            else:
                logger.error(f"❌ Model validation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model validation error: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        try:
            # Load config
            config_path = self.model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Calculate model size
            total_size = 0
            for file in self.model_path.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
            
            model_info = {
                "model_type": config.get("model_type", "smollm3"),
                "architectures": config.get("architectures", ["SmolLM3ForCausalLM"]),
                "model_size_gb": total_size / (1024**3),
                "vocab_size": config.get("vocab_size", 32000),
                "hidden_size": config.get("hidden_size", 2048),
                "num_attention_heads": config.get("num_attention_heads", 16),
                "num_hidden_layers": config.get("num_hidden_layers", 24),
                "max_position_embeddings": config.get("max_position_embeddings", 8192)
            }
            
            logger.info(f"📊 Model info: {model_info}")
            return model_info
            
        except Exception as e:
            logger.error(f"❌ Failed to get model info: {e}")
            return {}
    
    def run_quantization(self, repo_name: str, quant_type: str = "int8_weight_only") -> bool:
        """Run quantization on the model"""
        try:
            logger.info(f"🔄 Running quantization: {quant_type}")
            
            # Check if quantization script exists
            quantize_script = Path("scripts/model_tonic/quantize_model.py")
            if not quantize_script.exists():
                logger.error(f"❌ Quantization script not found: {quantize_script}")
                return False
            
            # Run quantization
            cmd = [
                sys.executable, str(quantize_script),
                str(self.model_path),
                repo_name,
                "--quant-type", quant_type,
                "--device", "auto"
            ]
            
            if self.hf_token:
                cmd.extend(["--token", self.hf_token])
            
            logger.info(f"🚀 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                logger.info("✅ Quantization completed successfully")
                logger.info(result.stdout)
                return True
            else:
                logger.error("❌ Quantization failed")
                logger.error(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Quantization timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to run quantization: {e}")
            return False
    
    def run_model_push(self, repo_name: str) -> bool:
        """Push the model to HF Hub"""
        try:
            logger.info(f"🔄 Pushing model to: {repo_name}")
            
            # Check if push script exists
            push_script = Path("scripts/model_tonic/push_to_huggingface.py")
            if not push_script.exists():
                logger.error(f"❌ Push script not found: {push_script}")
                return False
            
            # Run push
            cmd = [
                sys.executable, str(push_script),
                str(self.model_path),
                repo_name
            ]
            
            if self.hf_token:
                cmd.extend(["--token", self.hf_token])
            
            logger.info(f"🚀 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                logger.info("✅ Model push completed successfully")
                logger.info(result.stdout)
                return True
            else:
                logger.error("❌ Model push failed")
                logger.error(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Model push timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to push model: {e}")
            return False
    
    def process_model(self, repo_name: str, quantize: bool = True, push: bool = True) -> bool:
        """Complete model processing workflow"""
        logger.info("🚀 Starting model processing...")
        
        # Step 1: Validate model
        if not self.validate_model():
            logger.error("❌ Model validation failed")
            return False
        
        # Step 2: Get model info
        model_info = self.get_model_info()
        
        # Step 3: Quantize if requested
        if quantize:
            if not self.run_quantization(repo_name):
                logger.error("❌ Quantization failed")
                return False
        
        # Step 4: Push if requested
        if push:
            if not self.run_model_push(repo_name):
                logger.error("❌ Model push failed")
                return False
        
        logger.info("🎉 Model processing completed successfully!")
        logger.info(f"🌐 View your model at: https://huggingface.co/{repo_name}")
        
        return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process recovered model")
    parser.add_argument("repo_name", help="Hugging Face repository name (username/model-name)")
    parser.add_argument("--model-path", default="recovered_model", help="Path to recovered model")
    parser.add_argument("--no-quantize", action="store_true", help="Skip quantization")
    parser.add_argument("--no-push", action="store_true", help="Skip pushing to HF Hub")
    parser.add_argument("--quant-type", default="int8_weight_only", 
                       choices=["int8_weight_only", "int4_weight_only", "int8_dynamic"],
                       help="Quantization type")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ModelProcessor(args.model_path)
    
    # Process model
    success = processor.process_model(
        repo_name=args.repo_name,
        quantize=not args.no_quantize,
        push=not args.no_push
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())