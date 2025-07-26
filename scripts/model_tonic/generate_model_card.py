#!/usr/bin/env python3
"""
Generate unified model card from template
Handles template variables and conditional sections for quantized models
"""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelCardGenerator:
    """Generate unified model cards from templates"""
    
    def __init__(self, template_path: str = "templates/model_card.md"):
        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.template_path}")
    
    def load_template(self) -> str:
        """Load the model card template"""
        with open(self.template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def process_conditionals(self, content: str, variables: Dict[str, Any]) -> str:
        """Process conditional sections in the template"""
        # Handle {{#if variable}}...{{/if}} blocks
        pattern = r'\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}'
        
        def replace_conditional(match):
            variable_name = match.group(1)
            conditional_content = match.group(2)
            
            # Check if variable exists and is truthy
            if variable_name in variables and variables[variable_name]:
                return conditional_content
            else:
                return ""
        
        return re.sub(pattern, replace_conditional, content, flags=re.DOTALL)
    
    def replace_variables(self, content: str, variables: Dict[str, Any]) -> str:
        """Replace template variables with actual values"""
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            content = content.replace(placeholder, str(value))
        
        return content
    
    def generate_model_card(self, variables: Dict[str, Any]) -> str:
        """Generate the complete model card"""
        # Load template
        content = self.load_template()
        
        # Process conditionals first
        content = self.process_conditionals(content, variables)
        
        # Replace variables
        content = self.replace_variables(content, variables)
        
        return content
    
    def save_model_card(self, content: str, output_path: str) -> bool:
        """Save the generated model card"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"✅ Model card saved to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save model card: {e}")
            return False

def create_default_variables() -> Dict[str, Any]:
    """Create default variables for the model card"""
    return {
        "model_name": "SmolLM3 Fine-tuned Model",
        "model_description": "A fine-tuned version of SmolLM3-3B for improved text generation and conversation capabilities.",
        "repo_name": "your-username/model-name",
        "base_model": "HuggingFaceTB/SmolLM3-3B",
        "dataset_name": "OpenHermes-FR",
        "training_config_type": "Custom Configuration",
        "trainer_type": "SFTTrainer",
        "batch_size": "8",
        "gradient_accumulation_steps": "16",
        "learning_rate": "5e-6",
        "max_epochs": "3",
        "max_seq_length": "2048",
        "hardware_info": "GPU (H100/A100)",
        "experiment_name": "smollm3-experiment",
        "trackio_url": "https://trackio.space/experiment",
        "dataset_repo": "tonic/trackio-experiments",
        "dataset_size": "~80K samples",
        "dataset_format": "Chat format",
        "author_name": "Your Name",
        "model_name_slug": "smollm3-fine-tuned",
        "quantized_models": False,
        "dataset_sample_size": None,
        "training_loss": "N/A",
        "validation_loss": "N/A",
        "perplexity": "N/A"
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate unified model card")
    parser.add_argument("--template", default="templates/model_card.md", 
                       help="Path to model card template")
    parser.add_argument("--output", default="README.md", 
                       help="Output path for generated model card")
    parser.add_argument("--repo-name", required=True, 
                       help="Hugging Face repository name")
    parser.add_argument("--model-name", help="Model name")
    parser.add_argument("--experiment-name", help="Experiment name")
    parser.add_argument("--dataset-name", help="Dataset name")
    parser.add_argument("--training-config", help="Training configuration type")
    parser.add_argument("--trainer-type", help="Trainer type")
    parser.add_argument("--batch-size", help="Batch size")
    parser.add_argument("--learning-rate", help="Learning rate")
    parser.add_argument("--max-epochs", help="Maximum epochs")
    parser.add_argument("--max-seq-length", help="Maximum sequence length")
    parser.add_argument("--hardware-info", help="Hardware information")
    parser.add_argument("--trackio-url", help="Trackio URL")
    parser.add_argument("--dataset-repo", help="Dataset repository")
    parser.add_argument("--author-name", help="Author name")
    parser.add_argument("--quantized-models", action="store_true", 
                       help="Include quantized models")
    parser.add_argument("--dataset-sample-size", help="Dataset sample size")
    parser.add_argument("--training-loss", help="Training loss value")
    parser.add_argument("--validation-loss", help="Validation loss value")
    parser.add_argument("--perplexity", help="Perplexity value")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create generator
        generator = ModelCardGenerator(args.template)
        
        # Create variables dictionary
        variables = create_default_variables()
        
        # Override with command line arguments
        if args.repo_name:
            variables["repo_name"] = args.repo_name
        if args.model_name:
            variables["model_name"] = args.model_name
        if args.experiment_name:
            variables["experiment_name"] = args.experiment_name
        if args.dataset_name:
            variables["dataset_name"] = args.dataset_name
        if args.training_config:
            variables["training_config_type"] = args.training_config
        if args.trainer_type:
            variables["trainer_type"] = args.trainer_type
        if args.batch_size:
            variables["batch_size"] = args.batch_size
        if args.learning_rate:
            variables["learning_rate"] = args.learning_rate
        if args.max_epochs:
            variables["max_epochs"] = args.max_epochs
        if args.max_seq_length:
            variables["max_seq_length"] = args.max_seq_length
        if args.hardware_info:
            variables["hardware_info"] = args.hardware_info
        if args.trackio_url:
            variables["trackio_url"] = args.trackio_url
        if args.dataset_repo:
            variables["dataset_repo"] = args.dataset_repo
        if args.author_name:
            variables["author_name"] = args.author_name
        if args.quantized_models:
            variables["quantized_models"] = True
        if args.dataset_sample_size:
            variables["dataset_sample_size"] = args.dataset_sample_size
        if args.training_loss:
            variables["training_loss"] = args.training_loss
        if args.validation_loss:
            variables["validation_loss"] = args.validation_loss
        if args.perplexity:
            variables["perplexity"] = args.perplexity
        
        # Generate model card
        print("🔄 Generating model card...")
        content = generator.generate_model_card(variables)
        
        # Save model card
        if generator.save_model_card(content, args.output):
            print("✅ Model card generated successfully!")
            print(f"📄 Output: {args.output}")
        else:
            print("❌ Failed to generate model card")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error generating model card: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 