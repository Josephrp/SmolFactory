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
- tonic
- legml
{{#if quantized_models}}- quantized{{/if}}
pipeline_tag: text-generation
base_model: {{base_model}}
{{#if dataset_name}}
datasets:
- {{dataset_name}}
{{/if}}
{{#if quantized_models}}
model-index:
- name: {{model_name}}
  results:
  - task:
      type: text-generation
    dataset:
      name: {{dataset_name}}
      type: {{dataset_name}}
    metrics:
    - name: Training Loss
      type: loss
      value: "{{training_loss|default:'N/A'}}"
    - name: Validation Loss
      type: loss
      value: "{{validation_loss|default:'N/A'}}"
    - name: Perplexity
      type: perplexity
      value: "{{perplexity|default:'N/A'}}"
- name: {{model_name}} (int8 quantized)
  results:
  - task:
      type: text-generation
    dataset:
      name: {{dataset_name}}
      type: {{dataset_name}}
    metrics:
    - name: Memory Reduction
      type: memory_efficiency
      value: "~50%"
    - name: Inference Speed
      type: speed
      value: "Faster"
- name: {{model_name}} (int4 quantized)
  results:
  - task:
      type: text-generation
    dataset:
      name: {{dataset_name}}
      type: {{dataset_name}}
    metrics:
    - name: Memory Reduction
      type: memory_efficiency
      value: "~75%"
    - name: Inference Speed
      type: speed
      value: "Significantly Faster"
{{else}}
model-index:
- name: {{model_name}}
  results:
  - task:
      type: text-generation
    dataset:
      name: {{dataset_name}}
      type: {{dataset_name}}
    metrics:
    - name: Training Loss
      type: loss
      value: "{{training_loss|default:'N/A'}}"
    - name: Validation Loss
      type: loss
      value: "{{validation_loss|default:'N/A'}}"
    - name: Perplexity
      type: perplexity
      value: "{{perplexity|default:'N/A'}}"
{{/if}}
{{#if author_name}}
author: {{author_name}}
{{/if}}
{{#if experiment_name}}
experiment_name: {{experiment_name}}
{{/if}}
{{#if trackio_url}}
trackio_url: {{trackio_url}}
{{/if}}
{{#if dataset_repo}}
dataset_repo: {{dataset_repo}}
{{/if}}
{{#if hardware_info}}
hardware: "{{hardware_info}}"
{{/if}}
{{#if training_config_type}}
training_config: {{training_config_type}}
{{/if}}
{{#if trainer_type}}
trainer_type: {{trainer_type}}
{{/if}}
{{#if batch_size}}
batch_size: {{batch_size}}
{{/if}}
{{#if learning_rate}}
learning_rate: {{learning_rate}}
{{/if}}
{{#if max_epochs}}
max_epochs: {{max_epochs}}
{{/if}}
{{#if max_seq_length}}
max_seq_length: {{max_seq_length}}
{{/if}}
{{#if dataset_sample_size}}
dataset_sample_size: {{dataset_sample_size}}
{{/if}}
{{#if dataset_size}}
dataset_size: {{dataset_size}}
{{/if}}
{{#if dataset_format}}
dataset_format: {{dataset_format}}
{{/if}}
{{#if gradient_accumulation_steps}}
gradient_accumulation_steps: {{gradient_accumulation_steps}}
{{/if}}
---

# {{model_name}}

{{model_description}}

## Model Details

- **Base Model**: SmolLM3-3B
- **Model Type**: Causal Language Model
- **Languages**: English, French
- **License**: Apache 2.0
- **Fine-tuned**: Yes
{{#if quantized_models}}
- **Quantized Versions**: Available in subdirectories
{{/if}}

## Usage

### Main Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the main model
model = AutoModelForCausalLM.from_pretrained(
    "{{repo_name}}",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("{{repo_name}}")

# Generate text
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device.type)
output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training Information

### Training Configuration
- **Base Model**: {{base_model}}
- **Dataset**: {{dataset_name}}
- **Training Config**: {{training_config_type}}
- **Trainer Type**: {{trainer_type}}
{{#if dataset_sample_size}}
- **Dataset Sample Size**: {{dataset_sample_size}}
{{/if}}

### Training Parameters
- **Batch Size**: {{batch_size}}
- **Gradient Accumulation**: {{gradient_accumulation_steps}}
- **Learning Rate**: {{learning_rate}}
- **Max Epochs**: {{max_epochs}}
- **Sequence Length**: {{max_seq_length}}

### Training Infrastructure
- **Hardware**: {{hardware_info}}
- **Monitoring**: Trackio integration
- **Experiment**: {{experiment_name}}

## Model Architecture

This is a fine-tuned version of the SmolLM3-3B model with the following specifications:

- **Base Model**: SmolLM3-3B
- **Parameters**: ~3B
- **Context Length**: {{max_seq_length}}
- **Languages**: English, French
- **Architecture**: Transformer-based causal language model

## Performance

The model provides:
- **Text Generation**: High-quality text generation capabilities
- **Conversation**: Natural conversation abilities
- **Multilingual**: Support for English and French
{{#if quantized_models}}
- **Quantized Versions**: Optimized for different deployment scenarios
{{/if}}

## Limitations

1. **Context Length**: Limited by the model's maximum sequence length
2. **Bias**: May inherit biases from the training data
3. **Factual Accuracy**: May generate incorrect or outdated information
4. **Safety**: Should be used responsibly with appropriate safeguards
{{#if quantized_models}}
5. **Quantization**: Quantized versions may have slightly reduced accuracy
{{/if}}

## Training Data

The model was fine-tuned on:
- **Dataset**: {{dataset_name}}
- **Size**: {{dataset_size}}
- **Format**: {{dataset_format}}
- **Languages**: English, French

## Evaluation

The model was evaluated using:
- **Metrics**: Loss, perplexity, and qualitative assessment
- **Monitoring**: Real-time tracking via Trackio
- **Validation**: Regular validation during training

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{{model_name_slug}},
  title={{{{model_name}}}},
  author={{{author_name}}},
  year={2024},
  url={https://huggingface.co/{{repo_name}}}
}
```

## License

This model is licensed under the Apache 2.0 License.

## Acknowledgments

- **Base Model**: SmolLM3-3B by HuggingFaceTB
- **Training Framework**: PyTorch, Transformers, PEFT
- **Monitoring**: Trackio integration
- **Quantization**: torchao library

## Support

For questions and support:
- Open an issue on the Hugging Face repository
- Check the model documentation
- Review the training logs and configuration

## Repository Structure

```
{{repo_name}}/
├── README.md (this file)
├── config.json
├── pytorch_model.bin
├── tokenizer.json
└── tokenizer_config.json
```

## Usage Examples

### Text Generation
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{{repo_name}}")
tokenizer = AutoTokenizer.from_pretrained("{{repo_name}}")

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

### Advanced Usage
```python
# With generation parameters
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
```

## Monitoring and Tracking

This model was trained with comprehensive monitoring:
- **Trackio Space**: {{trackio_url}}
- **Experiment**: {{experiment_name}}
- **Dataset Repository**: https://huggingface.co/datasets/{{dataset_repo}}
- **Training Logs**: Available in the experiment data

## Deployment

### Requirements
```bash
pip install torch transformers accelerate
{{#if quantized_models}}
pip install torchao  # For quantized models
{{/if}}
```

### Hardware Requirements
- **Main Model**: GPU with 8GB+ VRAM recommended

## Changelog

- **v1.0.0**: Initial release with fine-tuned model