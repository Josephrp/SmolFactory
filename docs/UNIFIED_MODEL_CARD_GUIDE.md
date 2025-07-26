# Unified Model Card System Guide

## Overview

The unified model card system provides a template-based approach to generate comprehensive model cards that include information about both the main fine-tuned model and any quantized versions. This system ensures consistency across all model repositories and provides users with complete information about all available model variants.

## Architecture

### Template System

The system uses a template-based approach with the following components:

1. **Template File**: `templates/model_card.md` - Contains the master template with conditional sections
2. **Generator Script**: `scripts/model_tonic/generate_model_card.py` - Processes templates and variables
3. **Integration**: Updated push scripts that use the unified model card generator

### Key Features

- **Conditional Sections**: Template supports conditional rendering based on variables (e.g., quantized models)
- **Variable Substitution**: Dynamic content based on training configuration and results
- **Unified Repository Structure**: Single repository with subdirectories for quantized models
- **Comprehensive Documentation**: Complete usage examples and deployment information

## Template Structure

### Conditional Sections

The template uses Handlebars-style conditionals:

```markdown
{{#if quantized_models}}
### Quantized Models

This repository also includes quantized versions of the model for improved efficiency:

#### int8 Weight-Only Quantization (GPU Optimized)
```python
model = AutoModelForCausalLM.from_pretrained("{{repo_name}}/int8")
```
{{/if}}
```

### Template Variables

The template supports the following variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `model_name` | Display name of the model | "SmolLM3 Fine-tuned Model" |
| `model_description` | Brief description | "A fine-tuned version of SmolLM3-3B..." |
| `repo_name` | Hugging Face repository name | "username/model-name" |
| `base_model` | Original model name | "HuggingFaceTB/SmolLM3-3B" |
| `dataset_name` | Training dataset | "OpenHermes-FR" |
| `training_config_type` | Training configuration | "H100 Lightweight" |
| `trainer_type` | Trainer used | "SFTTrainer" |
| `batch_size` | Training batch size | "8" |
| `learning_rate` | Learning rate | "5e-6" |
| `max_epochs` | Number of epochs | "3" |
| `max_seq_length` | Maximum sequence length | "2048" |
| `hardware_info` | Hardware used | "GPU (H100/A100)" |
| `experiment_name` | Experiment name | "smollm3-experiment" |
| `trackio_url` | Trackio monitoring URL | "https://trackio.space/exp" |
| `dataset_repo` | HF Dataset repository | "tonic/trackio-experiments" |
| `quantized_models` | Boolean for quantized models | `true` or `false` |
| `author_name` | Model author | "Your Name" |

## Repository Structure

### Single Repository Approach

Instead of creating separate repositories for quantized models, the system now uses a single repository with subdirectories:

```
username/model-name/
├── README.md (unified model card)
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
├── int8/ (quantized model for GPU)
│   ├── README.md
│   ├── config.json
│   └── pytorch_model.bin
└── int4/ (quantized model for CPU)
    ├── README.md
    ├── config.json
    └── pytorch_model.bin
```

### Benefits

1. **Unified Documentation**: Single README with information about all model variants
2. **Easier Discovery**: Users find all model versions in one place
3. **Consistent Branding**: Single repository name and description
4. **Simplified Management**: One repository to maintain and update

## Usage

### Automatic Generation (via launch.sh)

The unified model card is automatically generated during the training pipeline:

```bash
# The launch script automatically generates the unified model card
./launch.sh
```

### Manual Generation

You can generate model cards manually using the generator script:

```bash
python scripts/model_tonic/generate_model_card.py \
    --repo-name "username/model-name" \
    --model-name "My Fine-tuned Model" \
    --experiment-name "my-experiment" \
    --dataset-name "OpenHermes-FR" \
    --training-config "H100 Lightweight" \
    --batch-size "8" \
    --learning-rate "5e-6" \
    --max-epochs "3" \
    --quantized-models \
    --output "README.md"
```

### Integration with Push Script

The push script automatically uses the unified model card generator:

```python
# In push_to_huggingface.py
def create_model_card(self, training_config: Dict[str, Any], results: Dict[str, Any]) -> str:
    """Create a comprehensive model card using the unified template"""
    try:
        from scripts.model_tonic.generate_model_card import ModelCardGenerator
        
        variables = {
            "model_name": f"{self.repo_name.split('/')[-1]} - Fine-tuned SmolLM3",
            "repo_name": self.repo_name,
            "quantized_models": False,  # Updated if quantized models are added
            # ... other variables
        }
        
        generator = ModelCardGenerator()
        return generator.generate_model_card(variables)
        
    except Exception as e:
        # Fallback to simple model card
        return self._create_simple_model_card()
```

## Quantization Integration

### Quantized Model Cards

When quantized models are created, the system:

1. **Updates Main Model Card**: Sets `quantized_models = True` and includes usage examples
2. **Creates Subdirectory Cards**: Generates specific README files for each quantized version
3. **Maintains Consistency**: All cards reference the same repository structure

### Quantization Types

The system supports:

- **int8_weight_only**: GPU optimized, ~50% memory reduction
- **int4_weight_only**: CPU optimized, ~75% memory reduction
- **int8_dynamic**: Dynamic quantization for flexibility

### Usage Examples

```python
# Main model
model = AutoModelForCausalLM.from_pretrained("username/model-name")

# int8 quantized (GPU)
model = AutoModelForCausalLM.from_pretrained("username/model-name/int8")

# int4 quantized (CPU)
model = AutoModelForCausalLM.from_pretrained("username/model-name/int4")
```

## Template Customization

### Adding New Sections

To add new sections to the template:

1. **Edit Template**: Modify `templates/model_card.md`
2. **Add Variables**: Update the generator script with new variables
3. **Update Integration**: Modify push scripts to pass new variables

### Example: Adding Performance Metrics

```markdown
{{#if performance_metrics}}
## Performance Metrics

- **BLEU Score**: {{bleu_score}}
- **ROUGE Score**: {{rouge_score}}
- **Perplexity**: {{perplexity}}
{{/if}}
```

### Conditional Logic

The template supports complex conditional logic:

```markdown
{{#if quantized_models}}
{{#if int8_available}}
### int8 Quantized Model
{{/if}}
{{#if int4_available}}
### int4 Quantized Model
{{/if}}
{{/if}}
```

## Best Practices

### Template Design

1. **Clear Structure**: Use consistent headings and organization
2. **Comprehensive Information**: Include all relevant model details
3. **Usage Examples**: Provide clear code examples
4. **Limitations**: Document model limitations and biases
5. **Citations**: Include proper citations and acknowledgments

### Variable Management

1. **Default Values**: Provide sensible defaults for all variables
2. **Validation**: Validate variable types and ranges
3. **Documentation**: Document all available variables
4. **Fallbacks**: Provide fallback content for missing variables

### Repository Organization

1. **Single Repository**: Use one repository per model family
2. **Clear Subdirectories**: Use descriptive subdirectory names
3. **Consistent Naming**: Follow consistent naming conventions
4. **Documentation**: Maintain comprehensive documentation

## Troubleshooting

### Common Issues

1. **Template Not Found**: Ensure `templates/model_card.md` exists
2. **Variable Errors**: Check that all required variables are provided
3. **Conditional Issues**: Verify conditional syntax and logic
4. **Import Errors**: Ensure all dependencies are installed

### Debugging

```bash
# Test template generation
python scripts/model_tonic/generate_model_card.py \
    --repo-name "test/model" \
    --output "test_readme.md" \
    --debug
```

### Validation

The system includes validation for:

- Template file existence
- Required variables
- Conditional syntax
- Output file permissions

## Future Enhancements

### Planned Features

1. **Multiple Template Support**: Support for different template types
2. **Advanced Conditionals**: More complex conditional logic
3. **Template Inheritance**: Base templates with extensions
4. **Auto-Detection**: Automatic detection of model features
5. **Custom Sections**: User-defined template sections

### Extensibility

The system is designed to be easily extensible:

- **Plugin Architecture**: Support for custom template processors
- **Variable Sources**: Multiple sources for template variables
- **Output Formats**: Support for different output formats
- **Integration Points**: Easy integration with other tools

## Conclusion

The unified model card system provides a comprehensive, maintainable approach to model documentation. By using templates and conditional sections, it ensures consistency while providing flexibility for different model configurations and quantization options.

The single repository approach with subdirectories simplifies model management and improves user experience by providing all model variants in one location with unified documentation. 