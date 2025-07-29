#!/usr/bin/env python3
"""
Quantize Trained Model using torchao
Supports int8 (GPU) and int4 (CPU) quantization with Hugging Face Hub integration
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import subprocess
import shutil
import platform

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
    from torchao.quantization import (
        Int8WeightOnlyConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig
    )
    from torchao.dtypes import Int4CPULayout
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False
    print("Warning: torchao not available. Install with: pip install torchao")

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

class ModelQuantizer:
    """Quantize models using torchao with HF Hub integration"""
    
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
                experiment_name=experiment_name or "model_quantization",
                trackio_url=trackio_url,
                enable_tracking=bool(trackio_url),
                hf_token=self.hf_token,
                dataset_repo=self.dataset_repo
            )
        
        logger.info(f"Initialized ModelQuantizer for {repo_name}")
        logger.info(f"Dataset repository: {self.dataset_repo}")
    
    def validate_model_path(self) -> bool:
        """Validate that the model path exists and contains required files"""
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
        
        missing_required = []
        for file in required_files:
            if not (self.model_path / file).exists():
                missing_required.append(file)
        
        # Check if at least one model file exists
        model_file_exists = any((self.model_path / file).exists() for file in model_files)
        if not model_file_exists:
            missing_required.extend(model_files)
        
        if missing_required:
            logger.error(f"❌ Missing required model files: {missing_required}")
            return False
        
        logger.info(f"✅ Model path validated: {self.model_path}")
        return True
    
    def create_quantization_config(self, quant_type: str, group_size: int = 128) -> TorchAoConfig:
        """Create torchao quantization configuration"""
        if not TORCHAO_AVAILABLE:
            raise ImportError("torchao is required. Install with: pip install torchao")
        
        if quant_type == "int8_weight_only":
            quant_config = Int8WeightOnlyConfig(group_size=group_size)
        elif quant_type == "int4_weight_only":
            # For int4, we need to specify CPU layout
            quant_config = Int4WeightOnlyConfig(group_size=group_size, layout=Int4CPULayout())
        elif quant_type == "int8_dynamic":
            quant_config = Int8DynamicActivationInt8WeightConfig()
        else:
            raise ValueError(f"Unsupported quantization type: {quant_type}")
        
        return TorchAoConfig(quant_type=quant_config)
    
    def get_optimal_device(self, quant_type: str) -> str:
        """Get optimal device for quantization type"""
        if quant_type == "int4_weight_only":
            # Int4 quantization works better on CPU
            return "cpu"
        elif quant_type == "int8_weight_only":
            # Int8 quantization works on GPU
            if torch.cuda.is_available():
                return "cuda"
            else:
                logger.warning("⚠️ CUDA not available, falling back to CPU for int8")
                return "cpu"
        else:
            return "auto"
    
    def quantize_model_alternative(
        self, 
        quant_type: str, 
        device: str = "auto",
        group_size: int = 128,
        save_dir: Optional[str] = None
    ) -> Optional[str]:
        """Alternative quantization using bitsandbytes for better compatibility"""
        try:
            logger.info(f"🔄 Attempting alternative quantization for: {quant_type}")
            
            # Import bitsandbytes if available
            try:
                import bitsandbytes as bnb
                from transformers import BitsAndBytesConfig
                BNB_AVAILABLE = True
            except ImportError:
                BNB_AVAILABLE = False
                logger.error("❌ bitsandbytes not available for alternative quantization")
                return None
            
            if not BNB_AVAILABLE:
                return None
            
            # Create bitsandbytes config
            if quant_type == "int8_weight_only":
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
            elif quant_type == "int4_weight_only":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                logger.error(f"❌ Unsupported quantization type for alternative method: {quant_type}")
                return None
            
            # Load model with bitsandbytes quantization
            quantized_model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            
            # Determine save directory
            if save_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = f"quantized_{quant_type}_bnb_{timestamp}"
            
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save quantized model
            logger.info(f"💾 Saving quantized model to: {save_path}")
            quantized_model.save_pretrained(save_path, safe_serialization=False)
            
            # Copy tokenizer files if they exist
            tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
            for file in tokenizer_files:
                src_file = self.model_path / file
                if src_file.exists():
                    shutil.copy2(src_file, save_path / file)
                    logger.info(f"📋 Copied {file}")
            
            logger.info(f"✅ Alternative quantization successful: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"❌ Alternative quantization failed: {e}")
            return None

    def quantize_model(
        self, 
        quant_type: str, 
        device: str = "auto",
        group_size: int = 128,
        save_dir: Optional[str] = None
    ) -> Optional[str]:
        """Quantize the model using torchao"""
        if not TORCHAO_AVAILABLE:
            logger.error("❌ torchao not available")
            return None
        
        try:
            logger.info(f"🔄 Loading model from: {self.model_path}")
            logger.info(f"🔄 Quantization type: {quant_type}")
            logger.info(f"🔄 Device: {device}")
            logger.info(f"🔄 Group size: {group_size}")
            
            # Determine optimal device
            if device == "auto":
                device = self.get_optimal_device(quant_type)
                logger.info(f"🔄 Using device: {device}")
            
            # Create quantization config
            quantization_config = self.create_quantization_config(quant_type, group_size)
            
            # Load model with appropriate device mapping
            if device == "cpu":
                device_map = "cpu"
                torch_dtype = torch.float32
            elif device == "cuda":
                device_map = "auto"
                torch_dtype = torch.bfloat16
            else:
                device_map = "auto"
                torch_dtype = "auto"
            
            # Load and quantize the model
            quantized_model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch_dtype,
                device_map=device_map,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True
            )
            
            # Determine save directory
            if save_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = f"quantized_{quant_type}_{timestamp}"
            
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save quantized model (don't use safetensors for torchao)
            logger.info(f"💾 Saving quantized model to: {save_path}")
            
            # For torchao models, we need to handle serialization carefully
            try:
                quantized_model.save_pretrained(save_path, safe_serialization=False)
            except Exception as save_error:
                logger.warning(f"⚠️ Standard save failed: {save_error}")
                logger.info("🔄 Attempting alternative save method...")
                
                # Try saving without quantization config
                try:
                    # Remove quantization config temporarily
                    original_config = quantized_model.config.quantization_config
                    quantized_model.config.quantization_config = None
                    quantized_model.save_pretrained(save_path, safe_serialization=False)
                    quantized_model.config.quantization_config = original_config
                except Exception as alt_save_error:
                    logger.error(f"❌ Alternative save also failed: {alt_save_error}")
                    return None
            
            # Copy tokenizer files if they exist
            tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
            for file in tokenizer_files:
                src_file = self.model_path / file
                if src_file.exists():
                    shutil.copy2(src_file, save_path / file)
                    logger.info(f"📋 Copied {file}")
            
            logger.info(f"✅ Model quantized successfully: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"❌ Quantization failed: {e}")
            # Try alternative quantization method
            logger.info("🔄 Attempting alternative quantization method...")
            return self.quantize_model_alternative(quant_type, device, group_size, save_dir)
    
    def create_quantized_model_card(self, quant_type: str, original_model: str, subdir: str) -> str:
        """Create a model card for the quantized model"""
        repo_name = self.repo_name
        card_content = f"""---
language:
- en
- fr
license: apache-2.0
tags:
- quantized
- {quant_type}
- smollm3
- fine-tuned
---

# Quantized SmolLM3 Model

This is a quantized version of the SmolLM3 model using torchao quantization.

## Model Details

- **Base Model**: SmolLM3-3B
- **Quantization Type**: {quant_type}
- **Original Model**: {original_model}
- **Quantization Library**: torchao
- **Hardware Compatibility**: {'GPU' if 'int8' in quant_type else 'CPU'}

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the quantized model
model = AutoModelForCausalLM.from_pretrained(
    f"{repo_name}/{subdir}",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(f"{repo_name}/{subdir}")

# Generate text
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device.type)
output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Quantization Details

- **Method**: torchao {quant_type}
- **Precision**: {'8-bit' if 'int8' in quant_type else '4-bit'}
- **Memory Reduction**: {'~50%' if 'int8' in quant_type else '~75%'}
- **Speed**: {'Faster inference with minimal accuracy loss' if 'int8' in quant_type else 'Significantly faster inference with some accuracy trade-off'}

## Training Information

This model was quantized from a fine-tuned SmolLM3 model using the torchao library.
The quantization process preserves the model's capabilities while reducing memory usage and improving inference speed.

## Limitations

- Quantized models may have slightly reduced accuracy compared to the original model
- {quant_type} quantization is optimized for {'GPU inference' if 'int8' in quant_type else 'CPU inference'}
- Some advanced features may not be available in quantized form

## Citation

If you use this model, please cite the original SmolLM3 paper and mention the quantization process.

```bibtex
@misc{{smollm3-quantized,
  title={{Quantized SmolLM3 Model}},
  author={{Your Name}},
  year={{2024}},
  url={{https://huggingface.co/{repo_name}/{subdir}}}
}}
```
"""
        return card_content
    
    def create_quantized_readme(self, quant_type: str, original_model: str, subdir: str) -> str:
        """Create a README for the quantized model repository"""
        repo_name = self.repo_name
        readme_content = f"""# Quantized SmolLM3 Model

This repository contains a quantized version of the SmolLM3 model using torchao quantization.

## Model Information

- **Model Type**: Quantized SmolLM3-3B
- **Quantization**: {quant_type}
- **Original Model**: {original_model}
- **Library**: torchao
- **Hardware**: {'GPU optimized' if 'int8' in quant_type else 'CPU optimized'}

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the quantized model
model = AutoModelForCausalLM.from_pretrained(
    f"{repo_name}/{subdir}",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(f"{repo_name}/{subdir}")

# Generate text
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device.type)
output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Quantization Benefits

- **Memory Efficiency**: {'~50% reduction in memory usage' if 'int8' in quant_type else '~75% reduction in memory usage'}
- **Speed**: {'Faster inference with minimal accuracy loss' if 'int8' in quant_type else 'Significantly faster inference'}
- **Compatibility**: {'GPU optimized for high-performance inference' if 'int8' in quant_type else 'CPU optimized for deployment'}

## Installation

```bash
pip install torchao transformers
```

## Usage Examples

### Text Generation
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(f"{repo_name}/{subdir}")
tokenizer = AutoTokenizer.from_pretrained(f"{repo_name}/{subdir}")

text = "The future of artificial intelligence is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Conversation
```python
def chat_with_model(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

response = chat_with_model("Hello, how are you today?")
print(response)
```

## Model Architecture

This is a quantized version of the SmolLM3-3B model with the following specifications:

- **Base Model**: SmolLM3-3B
- **Quantization**: {quant_type}
- **Parameters**: ~3B (quantized)
- **Context Length**: Variable (depends on original model)
- **Languages**: English, French

## Performance

The quantized model provides:

- **Memory Usage**: {'~50% of original model' if 'int8' in quant_type else '~25% of original model'}
- **Inference Speed**: {'Faster than original with minimal accuracy loss' if 'int8' in quant_type else 'Significantly faster with some accuracy trade-off'}
- **Accuracy**: {'Minimal degradation' if 'int8' in quant_type else 'Some degradation acceptable for speed'}

## Limitations

1. **Accuracy**: Quantized models may have slightly reduced accuracy
2. **Compatibility**: {'GPU optimized, may not work on CPU' if 'int8' in quant_type else 'CPU optimized, may not work on GPU'}
3. **Features**: Some advanced features may not be available
4. **Training**: Cannot be further fine-tuned in quantized form

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{smollm3-quantized,
  title={{Quantized SmolLM3 Model}},
  author={{Your Name}},
  year={{2024}},
  url={{https://huggingface.co/{repo_name}/{subdir}}}
}}
```

## License

This model is licensed under the Apache 2.0 License.

## Support

For questions and support, please open an issue on the Hugging Face repository.
"""
        return readme_content
    
    def push_quantized_model(
        self, 
        quantized_model_path: str, 
        quant_type: str,
        original_model: str
    ) -> bool:
        """Push quantized model to the same Hugging Face repository as the main model"""
        try:
            logger.info(f"🚀 Pushing quantized model to subdirectory in: {self.repo_name}")
            
            # Determine subdirectory name based on quantization type
            if quant_type == "int8_weight_only":
                subdir = "int8"
            elif quant_type == "int4_weight_only":
                subdir = "int4"
            elif quant_type == "int8_dynamic":
                subdir = "int8_dynamic"
            else:
                subdir = quant_type.replace("_", "-")
            
            # Create repository if it doesn't exist
            create_repo(
                repo_id=self.repo_name,
                token=self.token,
                private=self.private,
                exist_ok=True
            )
            
            # Create model card for the quantized version
            model_card = self.create_quantized_model_card(quant_type, original_model, subdir)
            model_card_path = Path(quantized_model_path) / "README.md"
            with open(model_card_path, 'w', encoding='utf-8') as f:
                f.write(model_card)
            
            # Upload all files to subdirectory
            logger.info(f"📤 Uploading quantized model files to {subdir}/ subdirectory...")
            for file_path in Path(quantized_model_path).rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(quantized_model_path)
                    # Upload to subdirectory within the repository
                    repo_path = f"{subdir}/{relative_path}"
                    upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=repo_path,
                        repo_id=self.repo_name,
                        token=self.token
                    )
                    logger.info(f"📤 Uploaded: {repo_path}")
            
            logger.info(f"✅ Quantized model pushed successfully to: https://huggingface.co/{self.repo_name}/{subdir}")
            
            # Log to Trackio if available
            if self.monitor:
                self.monitor.log_metric("quantization_type", quant_type)
                self.monitor.log_metric("quantized_model_url", f"https://huggingface.co/{self.repo_name}/{subdir}")
                self.monitor.log_artifact("quantized_model_path", quantized_model_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to push quantized model: {e}")
            return False
    
    def log_to_trackio(self, action: str, details: Dict[str, Any]):
        """Log quantization events to Trackio"""
        if self.monitor:
            try:
                # Use the correct monitoring method
                if hasattr(self.monitor, 'log_event'):
                    self.monitor.log_event(action, details)
                elif hasattr(self.monitor, 'log_metric'):
                    # Log as metric instead
                    self.monitor.log_metric(action, details.get('value', 1.0))
                elif hasattr(self.monitor, 'log'):
                    # Use generic log method
                    self.monitor.log(action, details)
                else:
                    # Just log locally if no monitoring method available
                    logger.info(f"📊 {action}: {details}")
                logger.info(f"📊 Logged to Trackio: {action}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to log to Trackio: {e}")
        else:
            # Log locally if no monitor available
            logger.info(f"📊 {action}: {details}")
    
    def quantize_and_push(
        self, 
        quant_type: str, 
        device: str = "auto",
        group_size: int = 128
    ) -> bool:
        """Complete quantization and push workflow"""
        try:
            # Validate model path
            if not self.validate_model_path():
                return False
            
            # Log start of quantization
            self.log_to_trackio("quantization_started", {
                "quant_type": quant_type,
                "device": device,
                "group_size": group_size,
                "model_path": str(self.model_path)
            })
            
            # Quantize model
            quantized_path = self.quantize_model(quant_type, device, group_size)
            if not quantized_path:
                return False
            
            # Log successful quantization
            self.log_to_trackio("quantization_completed", {
                "quantized_path": quantized_path,
                "quant_type": quant_type
            })
            
            # Push to HF Hub
            original_model = str(self.model_path)
            if not self.push_quantized_model(quantized_path, quant_type, original_model):
                return False
            
            # Log successful push
            self.log_to_trackio("quantized_model_pushed", {
                "repo_name": self.repo_name,
                "quant_type": quant_type
            })
            
            logger.info(f"🎉 Quantization and push completed successfully!")
            logger.info(f"📊 Model: https://huggingface.co/{self.repo_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Quantization and push failed: {e}")
            self.log_to_trackio("quantization_failed", {"error": str(e)})
            return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Quantize model using torchao")
    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument("repo_name", help="Hugging Face repository name")
    parser.add_argument("--quant-type", choices=["int8_weight_only", "int4_weight_only", "int8_dynamic"], 
                       default="int8_weight_only", help="Quantization type")
    parser.add_argument("--device", default="auto", help="Device for quantization (auto, cpu, cuda)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--token", help="Hugging Face token")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--trackio-url", help="Trackio URL for monitoring")
    parser.add_argument("--experiment-name", help="Experiment name for tracking")
    parser.add_argument("--dataset-repo", help="HF Dataset repository")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check torchao availability
    if not TORCHAO_AVAILABLE:
        logger.error("❌ torchao not available. Install with: pip install torchao")
        return 1
    
    # Initialize quantizer
    quantizer = ModelQuantizer(
        model_path=args.model_path,
        repo_name=args.repo_name,
        token=args.token,
        private=args.private,
        trackio_url=args.trackio_url,
        experiment_name=args.experiment_name,
        dataset_repo=args.dataset_repo
    )
    
    # Perform quantization and push
    success = quantizer.quantize_and_push(
        quant_type=args.quant_type,
        device=args.device,
        group_size=args.group_size
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 