#!/usr/bin/env python3
"""
Push Trained Model and Results to Hugging Face Hub
Integrates with Trackio monitoring and HF Datasets for complete model deployment
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import subprocess
import shutil

try:
    from huggingface_hub import HfApi, create_repo, upload_file
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")

try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from monitoring import SmolLM3Monitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("Warning: monitoring module not available")

logger = logging.getLogger(__name__)

class HuggingFacePusher:
    """Push trained models and results to Hugging Face Hub with HF Datasets integration"""
    
    def __init__(
        self,
        model_path: str,
        repo_name: str,
        token: Optional[str] = None,
        private: bool = False,
        trackio_url: Optional[str] = None,
        experiment_name: Optional[str] = None,
        dataset_repo: Optional[str] = None,
        hf_token: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        self.repo_name = repo_name
        self.token = token or hf_token or os.getenv('HF_TOKEN')
        self.private = private
        self.trackio_url = trackio_url
        self.experiment_name = experiment_name
        
        # HF Datasets configuration
        self.dataset_repo = dataset_repo or os.getenv('TRACKIO_DATASET_REPO', 'tonic/trackio-experiments')
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        
        # Initialize HF API
        if HF_AVAILABLE:
            self.api = HfApi(token=self.token)
        else:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
        
        # Initialize monitoring if available
        self.monitor = None
        if MONITORING_AVAILABLE:
            self.monitor = SmolLM3Monitor(
                experiment_name=experiment_name or "model_push",
                trackio_url=trackio_url,
                enable_tracking=bool(trackio_url),
                hf_token=self.hf_token,
                dataset_repo=self.dataset_repo
            )
        
        logger.info(f"Initialized HuggingFacePusher for {repo_name}")
        logger.info(f"Dataset repository: {self.dataset_repo}")
    
    def create_repository(self) -> bool:
        """Create the Hugging Face repository"""
        try:
            logger.info(f"Creating repository: {self.repo_name}")
            
            # Create repository
            create_repo(
                repo_id=self.repo_name,
                token=self.token,
                private=self.private,
                exist_ok=True
            )
            
            logger.info(f"✅ Repository created: https://huggingface.co/{self.repo_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create repository: {e}")
            return False
    
    def validate_model_path(self) -> bool:
        """Validate that the model path contains required files"""
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.model_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
        
        logger.info("✅ Model files validated")
        return True
    
    def create_model_card(self, training_config: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Create a comprehensive model card using the unified template"""
        try:
            # Import the model card generator
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
            from scripts.model_tonic.generate_model_card import ModelCardGenerator
            
            # Create variables for the template
            variables = {
                "model_name": f"{self.repo_name.split('/')[-1]} - Fine-tuned SmolLM3",
                "model_description": "A fine-tuned version of SmolLM3-3B for improved text generation and conversation capabilities.",
                "repo_name": self.repo_name,
                "base_model": "HuggingFaceTB/SmolLM3-3B",
                "dataset_name": training_config.get('dataset_name', 'OpenHermes-FR'),
                "training_config_type": training_config.get('training_config_type', 'Custom Configuration'),
                "trainer_type": training_config.get('trainer_type', 'SFTTrainer'),
                "batch_size": str(training_config.get('per_device_train_batch_size', 8)),
                "gradient_accumulation_steps": str(training_config.get('gradient_accumulation_steps', 16)),
                "learning_rate": str(training_config.get('learning_rate', '5e-6')),
                "max_epochs": str(training_config.get('num_train_epochs', 3)),
                "max_seq_length": str(training_config.get('max_seq_length', 2048)),
                "hardware_info": self._get_hardware_info(),
                "experiment_name": self.experiment_name or "smollm3-experiment",
                "trackio_url": self.trackio_url or "https://trackio.space/experiment",
                "dataset_repo": self.dataset_repo,
                "dataset_size": training_config.get('dataset_size', '~80K samples'),
                "dataset_format": training_config.get('dataset_format', 'Chat format'),
                "author_name": training_config.get('author_name', 'Your Name'),
                "model_name_slug": self.repo_name.split('/')[-1].lower().replace('-', '_'),
                "quantized_models": False,  # Will be updated if quantized models are added
                "dataset_sample_size": training_config.get('dataset_sample_size'),
                "training_loss": results.get('train_loss', 'N/A'),
                "validation_loss": results.get('eval_loss', 'N/A'),
                "perplexity": results.get('perplexity', 'N/A')
            }
            
            # Create generator and generate model card
            generator = ModelCardGenerator()
            return generator.generate_model_card(variables)
            
        except Exception as e:
            logger.error(f"Failed to generate model card from template: {e}")
            # Fallback to simple model card
            return f"""---
language:
- en
- fr
license: apache-2.0
tags:
- smollm3
- fine-tuned
- causal-lm
- text-generation
---

# {self.repo_name.split('/')[-1]}

This is a fine-tuned SmolLM3 model based on the HuggingFaceTB/SmolLM3-3B architecture.

## Model Details

- **Base Model**: HuggingFaceTB/SmolLM3-3B
- **Fine-tuning Method**: Supervised Fine-tuning
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Model Size**: {self._get_model_size():.1f} GB
- **Dataset Repository**: {self.dataset_repo}

## Training Configuration

```json
{json.dumps(training_config, indent=2)}
```

## Training Results

```json
{json.dumps(results, indent=2)}
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{self.repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_name}")

# Generate text
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Information

- **Base Model**: HuggingFaceTB/SmolLM3-3B
- **Hardware**: {self._get_hardware_info()}
- **Training Time**: {results.get('training_time_hours', 'Unknown')} hours
- **Final Loss**: {results.get('final_loss', 'Unknown')}
- **Final Accuracy**: {results.get('final_accuracy', 'Unknown')}
- **Dataset Repository**: {self.dataset_repo}

## Model Performance

- **Training Loss**: {results.get('train_loss', 'Unknown')}
- **Validation Loss**: {results.get('eval_loss', 'Unknown')}
- **Training Steps**: {results.get('total_steps', 'Unknown')}

## Experiment Tracking

This model was trained with experiment tracking enabled. Training metrics and configuration are stored in the HF Dataset repository: `{self.dataset_repo}`

## Limitations and Biases

This model is fine-tuned for specific tasks and may not generalize well to all use cases. Please evaluate the model's performance on your specific task before deployment.

## License

This model is licensed under the Apache 2.0 License.
"""
        # return model_card
    
    def _get_model_size(self) -> float:
        """Get model size in GB"""
        try:
            total_size = 0
            for file in self.model_path.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
            return total_size / (1024**3)  # Convert to GB
        except:
            return 0.0
    
    def _get_hardware_info(self) -> str:
        """Get hardware information"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                return f"GPU: {gpu_name}"
            else:
                return "CPU"
        except:
            return "Unknown"
    
    def upload_model_files(self) -> bool:
        """Upload model files to Hugging Face Hub"""
        try:
            logger.info("Uploading model files...")
            
            # Upload all files in the model directory
            for file_path in self.model_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.model_path)
                    remote_path = str(relative_path)
                    
                    logger.info(f"Uploading {relative_path}")
                    upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=remote_path,
                        repo_id=self.repo_name,
                        token=self.token
                    )
            
            logger.info("✅ Model files uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload model files: {e}")
            return False
    
    def upload_training_results(self, results_path: str) -> bool:
        """Upload training results and logs"""
        try:
            logger.info("Uploading training results...")
            
            results_files = [
                "train_results.json",
                "eval_results.json",
                "training_config.json",
                "training.log"
            ]
            
            for file_name in results_files:
                file_path = Path(results_path) / file_name
                if file_path.exists():
                    logger.info(f"Uploading {file_name}")
                    upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=f"training_results/{file_name}",
                        repo_id=self.repo_name,
                        token=self.token
                    )
            
            logger.info("✅ Training results uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload training results: {e}")
            return False
    
    def create_readme(self, training_config: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """Create and upload README.md"""
        try:
            logger.info("Creating README.md...")
            
            readme_content = f"""# {self.repo_name.split('/')[-1]}

A fine-tuned SmolLM3 model for text generation tasks.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_name}")

# Generate text
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Model Information

- **Base Model**: HuggingFaceTB/SmolLM3-3B
- **Fine-tuning Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Model Size**: {self._get_model_size():.1f} GB
- **Training Steps**: {results.get('total_steps', 'Unknown')}
- **Final Loss**: {results.get('final_loss', 'Unknown')}
- **Dataset Repository**: {self.dataset_repo}

## Training Configuration

```json
{json.dumps(training_config, indent=2)}
```

## Performance Metrics

```json
{json.dumps(results, indent=2)}
```

## Experiment Tracking

Training metrics and configuration are stored in the HF Dataset repository: `{self.dataset_repo}`

## Files

- `pytorch_model.bin`: Model weights
- `config.json`: Model configuration
- `tokenizer.json`: Tokenizer configuration
- `training_results/`: Training logs and results

## License

MIT License
"""
            
            # Write README to temporary file
            readme_path = Path("temp_readme.md")
            with open(readme_path, "w") as f:
                f.write(readme_content)
            
            # Upload README
            upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                token=self.token,
                repo_id=self.repo_name
            )
            
            # Clean up
            readme_path.unlink()
            
            logger.info("✅ README.md uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create README: {e}")
            return False
    
    def log_to_trackio(self, action: str, details: Dict[str, Any]):
        """Log push action to Trackio and HF Datasets"""
        if self.monitor:
            try:
                # Log to Trackio
                self.monitor.log_metrics({
                    "push_action": action,
                    "repo_name": self.repo_name,
                    "model_size_gb": self._get_model_size(),
                    "dataset_repo": self.dataset_repo,
                    **details
                })
                
                # Log training summary
                self.monitor.log_training_summary({
                    "model_push": True,
                    "model_repo": self.repo_name,
                    "dataset_repo": self.dataset_repo,
                    "push_date": datetime.now().isoformat(),
                    **details
                })
                
                logger.info(f"✅ Logged {action} to Trackio and HF Datasets")
            except Exception as e:
                logger.error(f"❌ Failed to log to Trackio: {e}")
    
    def push_model(self, training_config: Optional[Dict[str, Any]] = None, 
                   results: Optional[Dict[str, Any]] = None) -> bool:
        """Complete model push process with HF Datasets integration"""
        logger.info(f"🚀 Starting model push to {self.repo_name}")
        logger.info(f"📊 Dataset repository: {self.dataset_repo}")
        
        # Validate model path
        if not self.validate_model_path():
            return False
        
        # Create repository
        if not self.create_repository():
            return False
        
        # Load training config and results if not provided
        if training_config is None:
            training_config = self._load_training_config()
        
        if results is None:
            results = self._load_training_results()
        
        # Create and upload model card
        model_card = self.create_model_card(training_config, results)
        model_card_path = Path("temp_model_card.md")
        with open(model_card_path, "w") as f:
            f.write(model_card)
        
        try:
            upload_file(
                path_or_fileobj=str(model_card_path),
                path_in_repo="README.md",
                repo_id=self.repo_name,
                token=self.token
            )
        finally:
            model_card_path.unlink()
        
        # Upload model files
        if not self.upload_model_files():
            return False
        
        # Upload training results
        if results:
            self.upload_training_results(str(self.model_path))
        
        # Log to Trackio and HF Datasets
        self.log_to_trackio("model_push", {
            "model_path": str(self.model_path),
            "repo_name": self.repo_name,
            "private": self.private,
            "training_config": training_config,
            "results": results
        })
        
        logger.info(f"🎉 Model successfully pushed to: https://huggingface.co/{self.repo_name}")
        logger.info(f"📊 Experiment data stored in: {self.dataset_repo}")
        return True
    
    def _load_training_config(self) -> Dict[str, Any]:
        """Load training configuration"""
        config_path = self.model_path / "training_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return {"model_name": "HuggingFaceTB/SmolLM3-3B"}
    
    def _load_training_results(self) -> Dict[str, Any]:
        """Load training results"""
        results_path = self.model_path / "train_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                return json.load(f)
        return {"final_loss": "Unknown", "total_steps": "Unknown"}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Push trained model to Hugging Face Hub')
    
    # Required arguments
    parser.add_argument('model_path', type=str, help='Path to trained model directory')
    parser.add_argument('repo_name', type=str, help='Hugging Face repository name (username/repo-name)')
    
    # Optional arguments
    parser.add_argument('--token', type=str, default=None, help='Hugging Face token')
    parser.add_argument('--hf-token', type=str, default=None, help='Hugging Face token (alternative to --token)')
    parser.add_argument('--private', action='store_true', help='Make repository private')
    parser.add_argument('--trackio-url', type=str, default=None, help='Trackio Space URL for logging')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name for Trackio')
    parser.add_argument('--dataset-repo', type=str, default=None, help='HF Dataset repository for experiment storage')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting model push to Hugging Face Hub")
    
    # Initialize pusher
    try:
        pusher = HuggingFacePusher(
            model_path=args.model_path,
            repo_name=args.repo_name,
            token=args.token,
            private=args.private,
            trackio_url=args.trackio_url,
            experiment_name=args.experiment_name,
            dataset_repo=args.dataset_repo,
            hf_token=args.hf_token
        )
        
        # Push model
        success = pusher.push_model()
        
        if success:
            logger.info("✅ Model push completed successfully!")
            logger.info(f"🌐 View your model at: https://huggingface.co/{args.repo_name}")
            if args.dataset_repo:
                logger.info(f"📊 View experiment data at: https://huggingface.co/datasets/{args.dataset_repo}")
        else:
            logger.error("❌ Model push failed!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error during model push: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 