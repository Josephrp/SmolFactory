#!/usr/bin/env python3
"""
GPT-OSS Model Push Script
Specialized script for pushing GPT-OSS models to Hugging Face Hub
Handles LoRA weight merging and model card generation
"""

import os
import sys
import argparse
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def merge_lora_weights(checkpoint_path, base_model_name, output_path):
    """Merge LoRA weights with base model for inference"""
    
    print(f"Loading base model: {base_model_name}")
    
    # Load base model
    model_kwargs = {
        "attn_implementation": "eager", 
        "torch_dtype": "auto", 
        "use_cache": True, 
        "device_map": "auto"
    }
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs).cuda()
    
    print(f"Loading LoRA weights from: {checkpoint_path}")
    
    # Load and merge LoRA weights
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)
    
    return model, tokenizer

def create_gpt_oss_model_card(model_name, experiment_name, trackio_url, dataset_repo, author_name, model_description, training_config_type=None, dataset_name=None, batch_size=None, learning_rate=None, max_epochs=None, max_seq_length=None, trainer_type=None):
    """Create a comprehensive model card for GPT-OSS models using generate_model_card.py"""
    
    try:
        # Import the model card generator
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from generate_model_card import ModelCardGenerator, create_default_variables
        
        # Create generator
        generator = ModelCardGenerator()
        
        # Create variables for the model card
        variables = create_default_variables()
        
        # Update with GPT-OSS specific values
        variables.update({
            "repo_name": model_name,
            "model_name": model_name.split('/')[-1],
            "experiment_name": experiment_name or "gpt_oss_finetune",
            "dataset_repo": dataset_repo,
            "author_name": author_name or "GPT-OSS Fine-tuner",
            "model_description": model_description or "A fine-tuned version of OpenAI's GPT-OSS-20B model for multilingual reasoning tasks.",
            "training_config_type": training_config_type or "GPT-OSS Configuration",
            "base_model": "openai/gpt-oss-20b",
            "dataset_name": dataset_name or "HuggingFaceH4/Multilingual-Thinking",
            "trainer_type": trainer_type or "SFTTrainer",
            "batch_size": str(batch_size) if batch_size else "4",
            "learning_rate": str(learning_rate) if learning_rate else "2e-4",
            "max_epochs": str(max_epochs) if max_epochs else "1",
            "max_seq_length": str(max_seq_length) if max_seq_length else "2048",
            "hardware_info": "GPU (H100/A100)",
            "trackio_url": trackio_url or "N/A",
            "training_loss": "N/A",
            "validation_loss": "N/A",
            "perplexity": "N/A",
            "quantized_models": False
        })
        
        # Generate the model card
        model_card_content = generator.generate_model_card(variables)
        
        print("✅ Model card generated using generate_model_card.py")
        return model_card_content
        
    except Exception as e:
        print(f"❌ Failed to generate model card with generator: {e}")
        print("🔄 Falling back to original GPT-OSS model card")
        return _create_original_gpt_oss_model_card(model_name, experiment_name, trackio_url, dataset_repo, author_name, model_description)

def _create_original_gpt_oss_model_card(model_name, experiment_name, trackio_url, dataset_repo, author_name, model_description):
    """Create the original GPT-OSS model card as fallback"""
    
    card_content = f"""---
language:
- en
- es
- fr
- it
- de
- zh
- hi
- ja
- ko
- ar
license: mit
tags:
- gpt-oss
- multilingual
- reasoning
- chain-of-thought
- fine-tuned
---

# {model_name}

## Model Description

{model_description}

This model is a fine-tuned version of OpenAI's GPT-OSS-20B model, optimized for multilingual reasoning tasks. It has been trained on the Multilingual-Thinking dataset to generate chain-of-thought reasoning in multiple languages.

## Training Details

- **Base Model**: openai/gpt-oss-20b
- **Training Dataset**: HuggingFaceH4/Multilingual-Thinking
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: MXFP4
- **Experiment**: {experiment_name}
- **Monitoring**: {trackio_url}

## Usage

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
model = AutoModelForCausalLM.from_pretrained("{model_name}")

# Example: Reasoning in Spanish
messages = [
    {{"role": "system", "content": "reasoning language: Spanish"}},
    {{"role": "user", "content": "What is the capital of Australia?"}}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

output_ids = model.generate(input_ids, max_new_tokens=512)
response = tokenizer.batch_decode(output_ids)[0]
print(response)
```

### Multilingual Reasoning

The model supports reasoning in multiple languages:

- English
- Spanish (Español)
- French (Français)
- Italian (Italiano)
- German (Deutsch)
- Chinese (中文)
- Hindi (हिन्दी)
- Japanese (日本語)
- Korean (한국어)
- Arabic (العربية)

### System Prompt Format

To control the reasoning language, use the system prompt:

```
reasoning language: [LANGUAGE]
```

Example:
```
reasoning language: German
```

## Training Configuration

- **LoRA Rank**: 8
- **LoRA Alpha**: 16
- **Target Modules**: all-linear
- **Learning Rate**: 2e-4
- **Batch Size**: 4
- **Sequence Length**: 2048
- **Mixed Precision**: bf16

## Dataset Information

The model was trained on the Multilingual-Thinking dataset, which contains 1,000 examples of chain-of-thought reasoning translated into multiple languages.

## Limitations

- The model is designed for reasoning tasks and may not perform optimally on other tasks
- Reasoning quality may vary across languages
- The model inherits limitations from the base GPT-OSS-20B model

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{{model_name.replace("/", "_").replace("-", "_")},
  author = {{{author_name}}},
  title = {{{model_name}}},
  year = {{{datetime.now().year}}},
  publisher = {{Hugging Face}},
  journal = {{Hugging Face repository}},
  howpublished = {{\\url{{https://huggingface.co/{model_name}}}}}
  }}
```

## License

This model is licensed under the MIT License.

## Training Resources

- **Training Dataset**: https://huggingface.co/datasets/{dataset_repo}
- **Training Monitoring**: {trackio_url}
- **Base Model**: https://huggingface.co/openai/gpt-oss-20b

## Model Information

- **Architecture**: GPT-OSS-20B with LoRA adapters
- **Parameters**: 20B base + LoRA adapters
- **Context Length**: 2048 tokens
- **Languages**: 10+ languages supported
- **Task**: Multilingual reasoning and chain-of-thought generation
"""
    
    return card_content

def _resolve_repo_id(repo_name: str, hf_token: str) -> str:
    """Resolve to username/repo if only repo name was provided."""
    try:
        if "/" in repo_name:
            return repo_name
        from huggingface_hub import HfApi
        username = None
        if hf_token:
            try:
                api = HfApi(token=hf_token)
                info = api.whoami()
                username = info.get("name") or info.get("username")
            except Exception:
                username = None
        if not username:
            username = os.getenv("HF_USERNAME")
        if not username:
            raise ValueError("Could not determine HF username. Set HF_USERNAME or pass username/repo.")
        return f"{username}/{repo_name}"
    except Exception:
        return repo_name

def push_gpt_oss_model(checkpoint_path, repo_name, hf_token, trackio_url, experiment_name, dataset_repo, author_name, model_description, training_config_type=None, model_name=None, dataset_name=None, batch_size=None, learning_rate=None, max_epochs=None, max_seq_length=None, trainer_type=None):
    """Push GPT-OSS model to Hugging Face Hub"""
    
    print("=== GPT-OSS Model Push Pipeline ===")
    print(f"Checkpoint: {checkpoint_path}")
    full_repo_id = _resolve_repo_id(repo_name, hf_token)
    print(f"Repository: {full_repo_id}")
    print(f"Experiment: {experiment_name}")
    print(f"Author: {author_name}")
    
    # Validate checkpoint path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    
    # Create temporary directory for merged model
    temp_output = f"/tmp/gpt_oss_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(temp_output, exist_ok=True)
    
    try:
        # Merge LoRA weights with base model
        print("Merging LoRA weights with base model...")
        model, tokenizer = merge_lora_weights(
            checkpoint_path=checkpoint_path,
            base_model_name="openai/gpt-oss-20b",
            output_path=temp_output
        )
        
        # Create model card
        print("Creating model card...")
        model_card_content = create_gpt_oss_model_card(
            model_name=full_repo_id,
            experiment_name=experiment_name,
            trackio_url=trackio_url,
            dataset_repo=dataset_repo,
            author_name=author_name,
            model_description=model_description,
            training_config_type=training_config_type,
            dataset_name=dataset_name,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_seq_length=max_seq_length,
            trainer_type=trainer_type
        )
        
        # Save model card
        model_card_path = os.path.join(temp_output, "README.md")
        with open(model_card_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)
        
        # Push to Hugging Face Hub
        print(f"Pushing model to: {full_repo_id}")
        
        # Set HF token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        
        # Push using transformers
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(full_repo_id, private=False, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create repository: {e}")
        
        # Upload files
        print("Uploading model files...")
        api.upload_folder(
            folder_path=temp_output,
            repo_id=full_repo_id,
            repo_type="model"
        )
        
        print("✅ GPT-OSS model pushed successfully!")
        print(f"Model URL: https://huggingface.co/{full_repo_id}")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_output)
        
        return True
        
    except Exception as e:
        print(f"❌ Error pushing GPT-OSS model: {e}")
        
        # Clean up on error
        if os.path.exists(temp_output):
            import shutil
            shutil.rmtree(temp_output)
        
        return False

def main():
    parser = argparse.ArgumentParser(description="Push GPT-OSS model to Hugging Face Hub")
    parser.add_argument("checkpoint_path", help="Path to model checkpoint")
    parser.add_argument("repo_name", help="Hugging Face repository name")
    parser.add_argument("--token", required=True, help="Hugging Face token")
    parser.add_argument("--trackio-url", help="Trackio URL for model card")
    parser.add_argument("--experiment-name", help="Experiment name")
    parser.add_argument("--dataset-repo", help="Dataset repository")
    parser.add_argument("--author-name", help="Author name")
    parser.add_argument("--model-description", help="Model description")
    parser.add_argument("--training-config-type", help="Training configuration type")
    parser.add_argument("--model-name", help="Base model name")
    parser.add_argument("--dataset-name", help="Dataset name")
    parser.add_argument("--batch-size", help="Batch size")
    parser.add_argument("--learning-rate", help="Learning rate")
    parser.add_argument("--max-epochs", help="Maximum epochs")
    parser.add_argument("--max-seq-length", help="Maximum sequence length")
    parser.add_argument("--trainer-type", help="Trainer type")
    
    args = parser.parse_args()
    
    # Set defaults
    experiment_name = args.experiment_name or "gpt_oss_finetune"
    dataset_repo = args.dataset_repo or "HuggingFaceH4/Multilingual-Thinking"
    author_name = args.author_name or "GPT-OSS Fine-tuner"
    model_description = args.model_description or "A fine-tuned version of OpenAI's GPT-OSS-20B model for multilingual reasoning tasks."
    
    success = push_gpt_oss_model(
        checkpoint_path=args.checkpoint_path,
        repo_name=args.repo_name,
        hf_token=args.token,
        trackio_url=args.trackio_url,
        experiment_name=experiment_name,
        dataset_repo=dataset_repo,
        author_name=author_name,
        model_description=model_description,
        training_config_type=args.training_config_type,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        max_seq_length=args.max_seq_length,
        trainer_type=args.trainer_type
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 