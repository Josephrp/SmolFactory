# Model Card User Input Analysis

## Overview

This document analyzes the interaction between the model card template (`templates/model_card.md`), the model card generator (`scripts/model_tonic/generate_model_card.py`), and the launch script (`launch.sh`) to identify variables that require user input and improve the user experience.

## Template Variables Analysis

### Variables in `templates/model_card.md`

The model card template uses the following variables that can be populated with user input:

#### Core Model Information
- `{{model_name}}` - Display name of the model
- `{{model_description}}` - Brief description of the model
- `{{repo_name}}` - Hugging Face repository name
- `{{base_model}}` - Base model used for fine-tuning

#### Training Configuration
- `{{training_config_type}}` - Type of training configuration used
- `{{trainer_type}}` - Type of trainer (SFT, DPO, etc.)
- `{{batch_size}}` - Training batch size
- `{{gradient_accumulation_steps}}` - Gradient accumulation steps
- `{{learning_rate}}` - Learning rate used
- `{{max_epochs}}` - Maximum number of epochs
- `{{max_seq_length}}` - Maximum sequence length

#### Dataset Information
- `{{dataset_name}}` - Name of the dataset used
- `{{dataset_size}}` - Size of the dataset
- `{{dataset_format}}` - Format of the dataset
- `{{dataset_sample_size}}` - Sample size (for lightweight configs)

#### Training Results
- `{{training_loss}}` - Final training loss
- `{{validation_loss}}` - Final validation loss
- `{{perplexity}}` - Model perplexity

#### Infrastructure
- `{{hardware_info}}` - Hardware used for training
- `{{experiment_name}}` - Name of the experiment
- `{{trackio_url}}` - Trackio monitoring URL
- `{{dataset_repo}}` - HF Dataset repository

#### Author Information
- `{{author_name}}` - Author name for citations and attribution
- `{{model_name_slug}}` - URL-friendly model name

#### Quantization
- `{{quantized_models}}` - Boolean indicating if quantized models exist

## User Input Requirements

### Previously Missing User Inputs

#### 1. **Author Name** (`author_name`)
- **Purpose**: Used in model card metadata and citations
- **Template Usage**: `{{#if author_name}}author: {{author_name}}{{/if}}`
- **Citation Usage**: `author={{{author_name}}}`
- **Default**: "Your Name"
- **User Input Added**: ✅ **IMPLEMENTED**

#### 2. **Model Description** (`model_description`)
- **Purpose**: Brief description of the model's capabilities
- **Template Usage**: `{{model_description}}`
- **Default**: "A fine-tuned version of SmolLM3-3B for improved text generation and conversation capabilities."
- **User Input Added**: ✅ **IMPLEMENTED**

### Variables That Don't Need User Input

Most variables are automatically populated from:
- **Training Configuration**: Batch size, learning rate, epochs, etc.
- **System Detection**: Hardware info, model size, etc.
- **Auto-Generation**: Repository names, experiment names, etc.
- **Training Results**: Loss values, perplexity, etc.

## Implementation Changes

### 1. Launch Script Updates (`launch.sh`)

#### Added User Input Prompts
```bash
# Step 8.2: Author Information for Model Card
print_step "Step 8.2: Author Information"
echo "================================="

print_info "This information will be used in the model card and citation."
get_input "Author name for model card" "$HF_USERNAME" AUTHOR_NAME

print_info "Model description will be used in the model card and repository."
get_input "Model description" "A fine-tuned version of SmolLM3-3B for improved text generation and conversation capabilities." MODEL_DESCRIPTION
```

#### Updated Configuration Summary
```bash
echo "  Author: $AUTHOR_NAME"
```

#### Updated Model Push Call
```bash
python scripts/model_tonic/push_to_huggingface.py /output-checkpoint "$REPO_NAME" \
    --token "$HF_TOKEN" \
    --trackio-url "$TRACKIO_URL" \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-repo "$TRACKIO_DATASET_REPO" \
    --author-name "$AUTHOR_NAME" \
    --model-description "$MODEL_DESCRIPTION"
```

### 2. Push Script Updates (`scripts/model_tonic/push_to_huggingface.py`)

#### Added Command Line Arguments
```python
parser.add_argument('--author-name', type=str, default=None, help='Author name for model card')
parser.add_argument('--model-description', type=str, default=None, help='Model description for model card')
```

#### Updated Class Constructor
```python
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
    model_description: Optional[str] = None
):
```

#### Updated Model Card Generation
```python
variables = {
    "model_name": f"{self.repo_name.split('/')[-1]} - Fine-tuned SmolLM3",
    "model_description": self.model_description or "A fine-tuned version of SmolLM3-3B for improved text generation and conversation capabilities.",
    # ... other variables
    "author_name": self.author_name or training_config.get('author_name', 'Your Name'),
}
```

## User Experience Improvements

### 1. **Interactive Prompts**
- Users are now prompted for author name and model description
- Default values are provided for convenience
- Clear explanations of what each field is used for

### 2. **Configuration Summary**
- Author name is now displayed in the configuration summary
- Users can review all settings before proceeding

### 3. **Automatic Integration**
- User inputs are automatically passed to the model card generation
- No manual editing of scripts required

## Template Variable Categories

### Automatic Variables (No User Input Needed)
- `repo_name` - Auto-generated from username and date
- `base_model` - Always "HuggingFaceTB/SmolLM3-3B"
- `training_config_type` - From user selection
- `trainer_type` - From user selection
- `batch_size`, `learning_rate`, `max_epochs` - From training config
- `hardware_info` - Auto-detected
- `experiment_name` - Auto-generated with timestamp
- `trackio_url` - Auto-generated from space name
- `dataset_repo` - Auto-generated
- `training_loss`, `validation_loss`, `perplexity` - From training results

### User Input Variables (Now Implemented)
- `author_name` - ✅ **Added user prompt**
- `model_description` - ✅ **Added user prompt**

### Conditional Variables
- `quantized_models` - Set automatically based on quantization choices
- `dataset_sample_size` - Set based on training configuration type

## Benefits of These Changes

### 1. **Better Attribution**
- Author names are properly captured and used in citations
- Model cards include proper attribution

### 2. **Customizable Descriptions**
- Users can provide custom model descriptions
- Better model documentation and discoverability

### 3. **Improved User Experience**
- No need to manually edit scripts
- Interactive prompts with helpful defaults
- Clear feedback on what information is being collected

### 4. **Consistent Documentation**
- All model cards will have proper author information
- Standardized model descriptions
- Better integration with Hugging Face Hub

## Future Enhancements

### Potential Additional User Inputs
1. **License Selection** - Allow users to choose model license
2. **Model Tags** - Custom tags for better discoverability
3. **Usage Examples** - Custom usage examples for specific use cases
4. **Limitations Description** - Custom limitations based on training data

### Template Improvements
1. **Dynamic License** - Support for different license types
2. **Custom Tags** - User-defined model tags
3. **Usage Scenarios** - Template sections for different use cases

## Testing

The changes have been tested to ensure:
- ✅ Author name is properly passed to model card generation
- ✅ Model description is properly passed to model card generation
- ✅ Default values work correctly
- ✅ Configuration summary displays new fields
- ✅ Model push script accepts new parameters

## Conclusion

The analysis identified that the model card template had two key variables (`author_name` and `model_description`) that would benefit from user input. These have been successfully implemented with:

1. **Interactive prompts** in the launch script
2. **Command line arguments** in the push script
3. **Proper integration** with the model card generator
4. **User-friendly defaults** and clear explanations

This improves the overall user experience and ensures that model cards have proper attribution and descriptions. 