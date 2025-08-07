#!/usr/bin/env python3
"""
Push Trained Model and Results to Hugging Face Hub
Integrates with Trackio monitoring and HF Datasets for complete model deployment
"""

import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import subprocess
import shutil
import platform

# Set timeout for HF operations to prevent hanging
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
os.environ['HF_HUB_UPLOAD_TIMEOUT'] = '600'

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

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Operation timed out")

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
        hf_token: Optional[str] = None,
        author_name: Optional[str] = None,
        model_description: Optional[str] = None,
        training_config_type: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        batch_size: Optional[str] = None,
        learning_rate: Optional[str] = None,
        max_epochs: Optional[str] = None,
        max_seq_length: Optional[str] = None,
        trainer_type: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        # Original user input (may be just the repo name without username)
        self.repo_name = repo_name
        self.token = token or hf_token or os.getenv('HF_TOKEN')
        self.private = private
        self.trackio_url = trackio_url
        self.experiment_name = experiment_name
        self.author_name = author_name
        self.model_description = model_description
        
        # Training configuration details for model card generation
        self.training_config_type = training_config_type
        self.model_name = model_name  
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.max_seq_length = max_seq_length
        self.trainer_type = trainer_type
        
        # HF Datasets configuration
        self.dataset_repo = dataset_repo or os.getenv('TRACKIO_DATASET_REPO', 'tonic/trackio-experiments')
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        
        # Initialize HF API
        if HF_AVAILABLE:
            self.api = HfApi(token=self.token)
        else:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
        
        # Resolve the full repo id (username/repo) if user only provided repo name
        self.repo_id = self._resolve_repo_id(self.repo_name)

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
        
        logger.info(f"Initialized HuggingFacePusher for {self.repo_id}")
        logger.info(f"Dataset repository: {self.dataset_repo}")

    def _resolve_repo_id(self, repo_name: str) -> str:
        """Return a fully-qualified repo id in the form username/repo.

        If the provided name already contains a '/', it is returned unchanged.
        Otherwise, we attempt to derive the username from the authenticated token
        or from the HF_USERNAME environment variable.
        """
        try:
            if "/" in repo_name:
                return repo_name

            # Need a username. Prefer API whoami(), fallback to env HF_USERNAME
            username: Optional[str] = None
            if self.token:
                try:
                    user_info = self.api.whoami()
                    username = user_info.get("name") or user_info.get("username")
                except Exception:
                    username = None

            if not username:
                username = os.getenv("HF_USERNAME")

            if not username:
                raise ValueError(
                    "Username could not be determined. Provide a token or set HF_USERNAME, "
                    "or pass a fully-qualified repo id 'username/repo'."
                )

            return f"{username}/{repo_name}"
        except Exception as resolve_error:
            logger.error(f"Failed to resolve full repo id for '{repo_name}': {resolve_error}")
            # Fall back to provided value (may fail later at create/upload)
            return repo_name
    
    def create_repository(self) -> bool:
        """Create the Hugging Face repository"""
        try:
            logger.info(f"Creating repository: {self.repo_id}")
            
            # Create repository with timeout handling
            try:
                # Create repository
                create_repo(
                    repo_id=self.repo_id,
                    token=self.token,
                    private=self.private,
                    exist_ok=True
                )
                
                logger.info(f"✅ Repository created: https://huggingface.co/{self.repo_id}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Repository creation failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to create repository: {e}")
            return False
    
    def validate_model_path(self) -> bool:
        """Validate that the model path contains required files"""
        # Support both safetensors and pytorch formats
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
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
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
        
        logger.info("✅ Model files validated")
        return True
    
    def create_model_card(self, training_config: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Create a comprehensive model card using the generate_model_card.py script"""
        try:
            # Import the model card generator
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__)))
            from generate_model_card import ModelCardGenerator, create_default_variables
            
            # Create generator
            generator = ModelCardGenerator()
            
            # Create variables for the model card
            variables = create_default_variables()
            
            # Update with actual values
            variables.update({
                "repo_name": self.repo_id,
                "model_name": self.repo_id.split('/')[-1],
                "experiment_name": self.experiment_name or "model_push",
                "dataset_repo": self.dataset_repo,
                "author_name": self.author_name or "Model Author",
                "model_description": self.model_description or "A fine-tuned version of SmolLM3-3B for improved text generation capabilities.",
                "training_config_type": self.training_config_type or "Custom Configuration",
                "base_model": self.model_name or "HuggingFaceTB/SmolLM3-3B",
                "dataset_name": self.dataset_name or "Custom Dataset",
                "trainer_type": self.trainer_type or "SFTTrainer",
                "batch_size": str(self.batch_size) if self.batch_size else "8",
                "learning_rate": str(self.learning_rate) if self.learning_rate else "5e-6",
                "max_epochs": str(self.max_epochs) if self.max_epochs else "3",
                "max_seq_length": str(self.max_seq_length) if self.max_seq_length else "2048",
                "hardware_info": self._get_hardware_info(),
                "trackio_url": self.trackio_url or "N/A",
                "training_loss": str(results.get('train_loss', 'N/A')),
                "validation_loss": str(results.get('eval_loss', 'N/A')),
                "perplexity": str(results.get('perplexity', 'N/A')),
                "quantized_models": False  # Set to True if quantized models are available
            })
            
            # Generate the model card
            model_card_content = generator.generate_model_card(variables)
            
            logger.info("✅ Model card generated using generate_model_card.py")
            return model_card_content
            
        except Exception as e:
            logger.error(f"❌ Failed to generate model card with generator: {e}")
            logger.info("🔄 Falling back to simple model card")
            return self._create_simple_model_card(training_config, results)
    
    def _create_simple_model_card(self, training_config: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Create a simple model card without complex YAML to avoid formatting issues"""
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
pipeline_tag: text-generation
base_model: HuggingFaceTB/SmolLM3-3B
---

# {self.repo_id.split('/')[-1]}

This is a fine-tuned SmolLM3 model based on the HuggingFaceTB/SmolLM3-3B architecture.

## Model Details

- **Base Model**: HuggingFaceTB/SmolLM3-3B
- **Fine-tuning Method**: Supervised Fine-tuning
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Model Size**: {self._get_model_size():.1f} GB
- **Dataset Repository**: {self.dataset_repo}
- **Hardware**: {self._get_hardware_info()}

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
model = AutoModelForCausalLM.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")

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
        """Upload model files to Hugging Face Hub with timeout protection"""
        try:
            logger.info("Uploading model files...")
            
            # Upload all files in the model directory
            for file_path in self.model_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.model_path)
                    remote_path = str(relative_path)
                    
                    logger.info(f"Uploading {relative_path}")
                    
                    try:
                        upload_file(
                            path_or_fileobj=str(file_path),
                            path_in_repo=remote_path,
                            repo_id=self.repo_id,
                            token=self.token
                        )
                        logger.info(f"✅ Uploaded {relative_path}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to upload {relative_path}: {e}")
                        return False
            
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
                        repo_id=self.repo_id,
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
            
            readme_content = f"""# {self.repo_id.split('/')[-1]}

A fine-tuned SmolLM3 model for text generation tasks.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")

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

- `model.safetensors.index.json`: Model weights (safetensors format)
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
                repo_id=self.repo_id
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
                    "repo_name": self.repo_id,
                    "model_size_gb": self._get_model_size(),
                    "dataset_repo": self.dataset_repo,
                    **details
                })
                
                # Log training summary
                self.monitor.log_training_summary({
                    "model_push": True,
                    "model_repo": self.repo_id,
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
        logger.info(f"🚀 Starting model push to {self.repo_id}")
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
                repo_id=self.repo_id,
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
        
        logger.info(f"🎉 Model successfully pushed to: https://huggingface.co/{self.repo_id}")
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
    parser.add_argument('repo_name', type=str, help='Hugging Face repository name (repo-name). Username will be auto-detected from your token.')
    
    # Optional arguments
    parser.add_argument('--token', type=str, default=None, help='Hugging Face token')
    parser.add_argument('--hf-token', type=str, default=None, help='Hugging Face token (alternative to --token)')
    parser.add_argument('--private', action='store_true', help='Make repository private')
    parser.add_argument('--trackio-url', type=str, default=None, help='Trackio Space URL for logging')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name for Trackio')
    parser.add_argument('--dataset-repo', type=str, default=None, help='HF Dataset repository for experiment storage')
    parser.add_argument('--author-name', type=str, default=None, help='Author name for model card')
    parser.add_argument('--model-description', type=str, default=None, help='Model description for model card')
    parser.add_argument('--training-config-type', type=str, default=None, help='Training configuration type')
    parser.add_argument('--model-name', type=str, default=None, help='Base model name')
    parser.add_argument('--dataset-name', type=str, default=None, help='Dataset name')
    parser.add_argument('--batch-size', type=str, default=None, help='Batch size')
    parser.add_argument('--learning-rate', type=str, default=None, help='Learning rate')
    parser.add_argument('--max-epochs', type=str, default=None, help='Maximum epochs')
    parser.add_argument('--max-seq-length', type=str, default=None, help='Maximum sequence length')
    parser.add_argument('--trainer-type', type=str, default=None, help='Trainer type')
    
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
            hf_token=args.hf_token,
            author_name=args.author_name,
            model_description=args.model_description,
            training_config_type=args.training_config_type,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            max_seq_length=args.max_seq_length,
            trainer_type=args.trainer_type
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