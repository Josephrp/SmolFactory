#!/usr/bin/env python3
"""
Model Recovery and Deployment Script
Recovers trained model from cloud instance, quantizes it, and pushes to Hugging Face Hub
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


class ModelRecoveryPipeline:
    """Complete model recovery and deployment pipeline"""
    
    def __init__(
        self,
        model_path: str,
        repo_name: str,
        hf_token: Optional[str] = None,
        private: bool = False,
        quantize: bool = True,
        quant_types: Optional[list] = None,
        trackio_url: Optional[str] = None,
        experiment_name: Optional[str] = None,
        dataset_repo: Optional[str] = None,
        author_name: Optional[str] = None,
        model_description: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        self.repo_name = repo_name
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.private = private
        self.quantize = quantize
        self.quant_types = quant_types or ["int8_weight_only", "int4_weight_only"]
        self.trackio_url = trackio_url
        self.experiment_name = experiment_name
        self.dataset_repo = dataset_repo
        self.author_name = author_name
        self.model_description = model_description
        
        # Validate HF token
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable or --hf-token argument is required")
        
        logger.info(f"Initialized ModelRecoveryPipeline for {repo_name}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Quantization enabled: {self.quantize}")
        if self.quantize:
            logger.info(f"Quantization types: {self.quant_types}")
    
    def validate_model_path(self) -> bool:
        """Validate that the model path contains required files"""
        if not self.model_path.exists():
            logger.error(f"❌ Model path does not exist: {self.model_path}")
            return False
        
        # Check for essential model files
        required_files = ['config.json']
        
        # Check for model files (either safetensors or pytorch)
        model_files = [
            "model.safetensors.index.json",  # Safetensors format
            "pytorch_model.bin"  # PyTorch format
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.model_path / file).exists():
                missing_files.append(file)
        
        # Check if at least one model file exists
        model_file_exists = any((self.model_path / file).exists() for file in model_files)
        if not model_file_exists:
            missing_files.extend(model_files)
        
        if missing_files:
            logger.error(f"❌ Missing required model files: {missing_files}")
            return False
        
        logger.info("✅ Model files validated")
        return True
    
    def load_training_config(self) -> Dict[str, Any]:
        """Load training configuration from model directory"""
        config_files = [
            "training_config.json",
            "config_petite_llm_3_fr_1_20250727_152504.json",
            "config_petite_llm_3_fr_1_20250727_152524.json"
        ]
        
        for config_file in config_files:
            config_path = self.model_path / config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"✅ Loaded training config from: {config_file}")
                return config
        
        # Fallback to basic config
        logger.warning("⚠️ No training config found, using default")
        return {
            "model_name": "HuggingFaceTB/SmolLM3-3B",
            "dataset_name": "OpenHermes-FR",
            "training_config_type": "Custom Configuration",
            "trainer_type": "SFTTrainer",
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 16,
            "learning_rate": "5e-6",
            "num_train_epochs": 3,
            "max_seq_length": 2048,
            "dataset_size": "~80K samples",
            "dataset_format": "Chat format"
        }
    
    def load_training_results(self) -> Dict[str, Any]:
        """Load training results from model directory"""
        results_files = [
            "train_results.json",
            "training_summary_petite_llm_3_fr_1_20250727_152504.json",
            "training_summary_petite_llm_3_fr_1_20250727_152524.json"
        ]
        
        for results_file in results_files:
            results_path = self.model_path / results_file
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                logger.info(f"✅ Loaded training results from: {results_file}")
                return results
        
        # Fallback to basic results
        logger.warning("⚠️ No training results found, using default")
        return {
            "final_loss": "Unknown",
            "total_steps": "Unknown",
            "train_loss": "Unknown",
            "eval_loss": "Unknown"
        }
    
    def push_main_model(self) -> bool:
        """Push the main model to Hugging Face Hub"""
        try:
            logger.info("🚀 Pushing main model to Hugging Face Hub...")
            
            # Import push script
            from scripts.model_tonic.push_to_huggingface import HuggingFacePusher
            
            # Load training data
            training_config = self.load_training_config()
            training_results = self.load_training_results()
            
            # Initialize pusher
            pusher = HuggingFacePusher(
                model_path=str(self.model_path),
                repo_name=self.repo_name,
                token=self.hf_token,
                private=self.private,
                trackio_url=self.trackio_url,
                experiment_name=self.experiment_name,
                dataset_repo=self.dataset_repo,
                hf_token=self.hf_token,
                author_name=self.author_name,
                model_description=self.model_description
            )
            
            # Push model
            success = pusher.push_model(training_config, training_results)
            
            if success:
                logger.info(f"✅ Main model pushed successfully to: https://huggingface.co/{self.repo_name}")
                return True
            else:
                logger.error("❌ Failed to push main model")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error pushing main model: {e}")
            return False
    
    def quantize_and_push_models(self) -> bool:
        """Quantize and push models to Hugging Face Hub"""
        if not self.quantize:
            logger.info("⏭️ Skipping quantization (disabled)")
            return True
        
        try:
            logger.info("🔄 Starting quantization and push process...")
            
            # Import quantization script
            from scripts.model_tonic.quantize_model import ModelQuantizer
            
            success_count = 0
            total_count = len(self.quant_types)
            
            for quant_type in self.quant_types:
                logger.info(f"🔄 Processing quantization type: {quant_type}")
                
                # Initialize quantizer
                quantizer = ModelQuantizer(
                    model_path=str(self.model_path),
                    repo_name=self.repo_name,
                    token=self.hf_token,
                    private=self.private,
                    trackio_url=self.trackio_url,
                    experiment_name=self.experiment_name,
                    dataset_repo=self.dataset_repo,
                    hf_token=self.hf_token
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
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error during quantization: {e}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete model recovery and deployment pipeline"""
        logger.info("🚀 Starting complete model recovery and deployment pipeline")
        
        # Step 1: Validate model path
        if not self.validate_model_path():
            logger.error("❌ Model validation failed")
            return False
        
        # Step 2: Push main model
        if not self.push_main_model():
            logger.error("❌ Main model push failed")
            return False
        
        # Step 3: Quantize and push models
        if not self.quantize_and_push_models():
            logger.warning("⚠️ Quantization failed, but main model was pushed successfully")
        
        logger.info("🎉 Model recovery and deployment pipeline completed!")
        logger.info(f"🌐 View your model at: https://huggingface.co/{self.repo_name}")
        
        return True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Recover and deploy trained model to Hugging Face Hub')
    
    # Required arguments
    parser.add_argument('model_path', type=str, help='Path to trained model directory')
    parser.add_argument('repo_name', type=str, help='Hugging Face repository name (username/repo-name)')
    
    # Optional arguments
    parser.add_argument('--hf-token', type=str, default=None, help='Hugging Face token')
    parser.add_argument('--private', action='store_true', help='Make repository private')
    parser.add_argument('--no-quantize', action='store_true', help='Skip quantization')
    parser.add_argument('--quant-types', nargs='+', 
                       choices=['int8_weight_only', 'int4_weight_only', 'int8_dynamic'],
                       default=['int8_weight_only', 'int4_weight_only'],
                       help='Quantization types to apply')
    parser.add_argument('--trackio-url', type=str, default=None, help='Trackio Space URL for logging')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name for Trackio')
    parser.add_argument('--dataset-repo', type=str, default=None, help='HF Dataset repository for experiment storage')
    parser.add_argument('--author-name', type=str, default=None, help='Author name for model card')
    parser.add_argument('--model-description', type=str, default=None, help='Model description for model card')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting model recovery and deployment pipeline")
    
    # Initialize pipeline
    try:
        pipeline = ModelRecoveryPipeline(
            model_path=args.model_path,
            repo_name=args.repo_name,
            hf_token=args.hf_token,
            private=args.private,
            quantize=not args.no_quantize,
            quant_types=args.quant_types,
            trackio_url=args.trackio_url,
            experiment_name=args.experiment_name,
            dataset_repo=args.dataset_repo,
            author_name=args.author_name,
            model_description=args.model_description
        )
        
        # Run complete pipeline
        success = pipeline.run_complete_pipeline()
        
        if success:
            logger.info("✅ Model recovery and deployment completed successfully!")
            return 0
        else:
            logger.error("❌ Model recovery and deployment failed!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error during model recovery: {e}")
        return 1

if __name__ == "__main__":
    exit(main())