# Enhanced Model Card Metadata System

## Overview

The enhanced model card system now includes comprehensive YAML metadata that follows the [Hugging Face Model Cards specification](https://huggingface.co/docs/hub/en/model-cards). This ensures maximum compatibility with the Hugging Face Hub and provides rich metadata for model discovery and usage.

## Metadata Structure

### Core Metadata Fields

The model card template now includes the following metadata fields:

```yaml
---
language:
- en
- fr
license: apache-2.0
library_name: transformers
tags:
- smollm3
- fine-tuned
- causal-lm
- text-generation
- quantized
- dataset:OpenHermes-FR
- config:H100 Lightweight
pipeline_tag: text-generation
base_model: HuggingFaceTB/SmolLM3-3B
datasets:
- OpenHermes-FR
---
```

### Conditional Metadata

The system supports conditional metadata based on model configuration:

#### Quantized Models
When quantized models are available, additional metadata is included:

```yaml
quantization_types:
- int8_weight_only
- int4_weight_only
```

#### Model Index (Evaluation Results)
The system automatically generates structured evaluation results:

```yaml
model-index:
- name: Model Name
  results:
  - task:
      type: text-generation
    dataset:
      name: OpenHermes-FR
      type: OpenHermes-FR
    metrics:
    - name: Training Loss
      type: loss
      value: "2.1"
    - name: Validation Loss
      type: loss
      value: "2.3"
    - name: Perplexity
      type: perplexity
      value: "9.8"
```

For quantized models, additional entries are included:

```yaml
- name: Model Name (int8 quantized)
  results:
  - task:
      type: text-generation
    dataset:
      name: OpenHermes-FR
      type: OpenHermes-FR
    metrics:
    - name: Memory Reduction
      type: memory_efficiency
      value: "~50%"
    - name: Inference Speed
      type: speed
      value: "Faster"
```

## Metadata Fields Explained

### Required Fields

| Field | Description | Example |
|-------|-------------|---------|
| `language` | Supported languages | `["en", "fr"]` |
| `license` | Model license | `"apache-2.0"` |
| `library_name` | Primary library | `"transformers"` |
| `tags` | Model tags for discovery | `["smollm3", "fine-tuned"]` |
| `pipeline_tag` | Task type | `"text-generation"` |
| `base_model` | Original model | `"HuggingFaceTB/SmolLM3-3B"` |

### Optional Fields

| Field | Description | Example |
|-------|-------------|---------|
| `datasets` | Training datasets | `["OpenHermes-FR"]` |
| `author` | Model author | `"Your Name"` |
| `experiment_name` | Experiment tracking | `"smollm3-experiment"` |
| `trackio_url` | Monitoring URL | `"https://trackio.space/exp"` |
| `hardware` | Training hardware | `"GPU (H100/A100)"` |
| `training_config` | Configuration type | `"H100 Lightweight"` |
| `trainer_type` | Trainer used | `"SFTTrainer"` |
| `batch_size` | Training batch size | `"8"` |
| `learning_rate` | Learning rate | `"5e-6"` |
| `max_epochs` | Number of epochs | `"3"` |
| `max_seq_length` | Sequence length | `"2048"` |
| `gradient_accumulation_steps` | Gradient accumulation | `"16"` |

### Training Results

| Field | Description | Example |
|-------|-------------|---------|
| `training_loss` | Final training loss | `"2.1"` |
| `validation_loss` | Final validation loss | `"2.3"` |
| `perplexity` | Model perplexity | `"9.8"` |

## Benefits of Enhanced Metadata

### 1. Improved Discovery
- **Filtering**: Users can filter models by dataset, configuration, or hardware
- **Search**: Enhanced search capabilities on the Hugging Face Hub
- **Tags**: Automatic tag generation for better categorization

### 2. Better Model Cards
- **Structured Data**: Evaluation results are displayed in widgets
- **Consistent Format**: Follows Hugging Face standards
- **Rich Information**: Comprehensive model information

### 3. Integration Benefits
- **Papers with Code**: Model index data can be indexed in leaderboards
- **API Compatibility**: Better integration with Hugging Face APIs
- **Automated Tools**: Support for automated model analysis

## Usage Examples

### Basic Model Card Generation

```bash
python scripts/model_tonic/generate_model_card.py \
    --repo-name "username/model-name" \
    --model-name "My Fine-tuned Model" \
    --dataset-name "OpenHermes-FR" \
    --training-config "H100 Lightweight" \
    --batch-size "8" \
    --learning-rate "5e-6" \
    --max-epochs "3" \
    --training-loss "2.1" \
    --validation-loss "2.3" \
    --perplexity "9.8" \
    --output "README.md"
```

### With Quantized Models

```bash
python scripts/model_tonic/generate_model_card.py \
    --repo-name "username/model-name" \
    --model-name "My Fine-tuned Model" \
    --dataset-name "OpenHermes-FR" \
    --training-config "H100 Lightweight" \
    --batch-size "8" \
    --learning-rate "5e-6" \
    --max-epochs "3" \
    --training-loss "2.1" \
    --validation-loss "2.3" \
    --perplexity "9.8" \
    --quantized-models \
    --output "README.md"
```

## Template Variables

The enhanced template supports all the original variables plus new metadata fields:

### New Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `training_loss` | Training loss value | `"N/A"` |
| `validation_loss` | Validation loss value | `"N/A"` |
| `perplexity` | Model perplexity | `"N/A"` |

### Conditional Metadata

The template automatically includes:

- **Dataset Information**: When `dataset_name` is provided
- **Quantization Types**: When `quantized_models` is `true`
- **Evaluation Results**: When training metrics are available
- **Hardware Information**: When `hardware_info` is provided

## Integration with Training Pipeline

### Automatic Metadata Generation

The push script automatically extracts metadata from:

1. **Training Configuration**: Batch size, learning rate, epochs, etc.
2. **Training Results**: Loss values, perplexity, etc.
3. **Model Information**: Base model, hardware, etc.
4. **Experiment Tracking**: Trackio URLs, experiment names

### Example Integration

```python
# In push_to_huggingface.py
variables = {
    "model_name": f"{self.repo_name.split('/')[-1]} - Fine-tuned SmolLM3",
    "repo_name": self.repo_name,
    "base_model": "HuggingFaceTB/SmolLM3-3B",
    "dataset_name": training_config.get('dataset_name', 'OpenHermes-FR'),
    "training_config_type": training_config.get('training_config_type', 'Custom Configuration'),
    "trainer_type": training_config.get('trainer_type', 'SFTTrainer'),
    "batch_size": str(training_config.get('per_device_train_batch_size', 8)),
    "learning_rate": str(training_config.get('learning_rate', '5e-6')),
    "max_epochs": str(training_config.get('num_train_epochs', 3)),
    "hardware_info": self._get_hardware_info(),
    "training_loss": results.get('train_loss', 'N/A'),
    "validation_loss": results.get('eval_loss', 'N/A'),
    "perplexity": results.get('perplexity', 'N/A'),
    "quantized_models": False  # Updated if quantized models are added
}
```

## Validation and Testing

### Metadata Validation

The system includes validation for:

- **Required Fields**: Ensures all required metadata is present
- **Format Validation**: Validates YAML syntax and structure
- **Value Ranges**: Checks for reasonable values in numeric fields
- **Conditional Logic**: Verifies conditional metadata is properly included

### Test Coverage

The test suite verifies:

- **Basic Metadata**: All required fields are present
- **Conditional Metadata**: Quantized model metadata is included when appropriate
- **Evaluation Results**: Model index data is properly structured
- **Template Processing**: Variable substitution works correctly

## Best Practices

### 1. Metadata Completeness
- Include all available training information
- Provide accurate evaluation metrics
- Use consistent naming conventions

### 2. Conditional Logic
- Only include relevant metadata
- Use conditional sections appropriately
- Provide fallback values for missing data

### 3. Validation
- Test metadata generation with various configurations
- Verify YAML syntax is correct
- Check that all variables are properly substituted

### 4. Documentation
- Document all available metadata fields
- Provide examples for each field type
- Include troubleshooting information

## Future Enhancements

### Planned Features

1. **Additional Metrics**: Support for more evaluation metrics
2. **Custom Metadata**: User-defined metadata fields
3. **Validation Rules**: Configurable validation rules
4. **Auto-Detection**: Automatic detection of model features
5. **Integration APIs**: Better integration with external tools

### Extensibility

The system is designed to be easily extensible:

- **New Fields**: Easy to add new metadata fields
- **Custom Validators**: Support for custom validation logic
- **Template Extensions**: Support for template inheritance
- **API Integration**: Easy integration with external APIs

## Conclusion

The enhanced model card metadata system provides comprehensive, standards-compliant metadata that maximizes compatibility with the Hugging Face Hub while providing rich information for model discovery and usage. The system automatically generates appropriate metadata based on model configuration and training results, ensuring consistency and completeness across all model repositories. 