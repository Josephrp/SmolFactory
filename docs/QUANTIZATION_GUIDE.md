# Model Quantization Guide

## Overview

This guide covers the quantization functionality integrated into the SmolLM3 fine-tuning pipeline. The system supports creating quantized versions of trained models using `torchao` and automatically uploading them to Hugging Face Hub in a unified repository structure.

## Repository Structure

With the updated pipeline, all models (main and quantized) are stored in a single repository:

```
your-username/model-name/
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

## Quantization Types

### int8 Weight-Only Quantization (GPU Optimized)
- **Memory Reduction**: ~50% compared to original model
- **Speed**: Faster inference with minimal accuracy loss
- **Hardware**: GPU optimized for high-performance inference
- **Use Case**: Production deployments with GPU resources

### int4 Weight-Only Quantization (CPU Optimized)
- **Memory Reduction**: ~75% compared to original model
- **Speed**: Significantly faster inference with some accuracy trade-off
- **Hardware**: CPU optimized for deployment
- **Use Case**: Edge deployment, CPU-only environments

## Integration with Pipeline

### Automatic Quantization

The quantization process is integrated into the main training pipeline:

1. **Training**: Model is trained using the standard pipeline
2. **Model Push**: Main model is pushed to Hugging Face Hub
3. **Quantization Options**: User is prompted to create quantized versions
4. **Quantized Models**: Quantized models are created and pushed to subdirectories
5. **Unified Documentation**: Single model card covers all versions

### Pipeline Integration

The quantization step is added to `launch.sh` after the main model push:

```bash
# Step 16.5: Quantization Options
print_step "Step 16.5: Model Quantization Options"
echo "=========================================="

print_info "Would you like to create quantized versions of your model?"
print_info "Quantization reduces model size and improves inference speed."

# Ask about quantization
get_input "Create quantized models? (y/n)" "y" "CREATE_QUANTIZED"

if [ "$CREATE_QUANTIZED" = "y" ] || [ "$CREATE_QUANTIZED" = "Y" ]; then
    print_info "Quantization options:"
    print_info "1. int8_weight_only (GPU optimized, ~50% memory reduction)"
    print_info "2. int4_weight_only (CPU optimized, ~75% memory reduction)"
    print_info "3. Both int8 and int4 versions"
    
    select_option "Select quantization type:" "int8_weight_only" "int4_weight_only" "both" "QUANT_TYPE"
    
    # Create quantized models in the same repository
    python scripts/model_tonic/quantize_model.py /output-checkpoint "$REPO_NAME" \
        --quant-type "$QUANT_TYPE" \
        --device "$DEVICE" \
        --token "$HF_TOKEN" \
        --trackio-url "$TRACKIO_URL" \
        --experiment-name "${EXPERIMENT_NAME}-${QUANT_TYPE}" \
        --dataset-repo "$TRACKIO_DATASET_REPO"
fi
```

## Standalone Quantization

### Using the Standalone Script

For models already uploaded to Hugging Face Hub:

```bash
python scripts/model_tonic/quantize_standalone.py \
    "your-username/model-name" \
    "your-username/model-name" \
    --quant-type "int8_weight_only" \
    --device "auto" \
    --token "your-hf-token"
```

### Command Line Options

```bash
python scripts/model_tonic/quantize_standalone.py model_path repo_name [options]

Options:
  --quant-type {int8_weight_only,int4_weight_only,int8_dynamic}
                        Quantization type (default: int8_weight_only)
  --device DEVICE       Device for quantization (auto, cpu, cuda)
  --group-size GROUP_SIZE
                        Group size for quantization (default: 128)
  --token TOKEN         Hugging Face token
  --private             Create private repository
  --trackio-url TRACKIO_URL
                        Trackio URL for monitoring
  --experiment-name EXPERIMENT_NAME
                        Experiment name for tracking
  --dataset-repo DATASET_REPO
                        HF Dataset repository
  --save-only           Save quantized model locally without pushing to HF
```

## Loading Quantized Models

### Loading Main Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the main model
model = AutoModelForCausalLM.from_pretrained(
    "your-username/model-name",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("your-username/model-name")
```

### Loading int8 Quantized Model (GPU)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load int8 quantized model (GPU optimized)
model = AutoModelForCausalLM.from_pretrained(
    "your-username/model-name/int8",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("your-username/model-name/int8")
```

### Loading int4 Quantized Model (CPU)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load int4 quantized model (CPU optimized)
model = AutoModelForCausalLM.from_pretrained(
    "your-username/model-name/int4",
    device_map="cpu",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("your-username/model-name/int4")
```

## Usage Examples

### Text Generation with Quantized Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load quantized model
model = AutoModelForCausalLM.from_pretrained("your-username/model-name/int8")
tokenizer = AutoTokenizer.from_pretrained("your-username/model-name/int8")

# Generate text
text = "The future of artificial intelligence is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Conversation with Quantized Model

```python
def chat_with_quantized_model(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

response = chat_with_quantized_model("Hello, how are you today?")
print(response)
```

## Configuration Options

### Quantization Parameters

- **group_size**: Group size for quantization (default: 128)
- **device**: Target device for quantization (auto, cpu, cuda)
- **quant_type**: Type of quantization to apply

### Hardware Requirements

- **Main Model**: GPU with 8GB+ VRAM recommended
- **int8 Model**: GPU with 4GB+ VRAM
- **int4 Model**: CPU deployment possible

## Performance Comparison

| Model Type | Memory Usage | Speed | Accuracy | Use Case |
|------------|--------------|-------|----------|----------|
| Original | 100% | Baseline | Best | Development, Research |
| int8 | ~50% | Faster | Minimal loss | Production GPU |
| int4 | ~25% | Fastest | Some loss | Edge, CPU deployment |

## Best Practices

### When to Use Quantization

1. **int8 (GPU)**: When you need faster inference with minimal accuracy loss
2. **int4 (CPU)**: When deploying to CPU-only environments or edge devices
3. **Both**: When you need flexibility for different deployment scenarios

### Memory Optimization

- Use int8 for GPU deployments with memory constraints
- Use int4 for CPU deployments or very memory-constrained environments
- Consider the trade-off between speed and accuracy

### Deployment Considerations

- Test quantized models on your specific use case
- Monitor performance and accuracy in production
- Consider using the main model for development and quantized versions for deployment

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use int8 quantization
2. **Import Errors**: Install torchao: `pip install torchao>=0.10.0`
3. **Model Loading Errors**: Ensure the model path is correct and accessible

### Debugging

```bash
# Test quantization functionality
python tests/test_quantization.py

# Check torchao installation
python -c "import torchao; print('torchao available')"

# Verify model files
ls -la /path/to/model/
```

## Monitoring and Tracking

### Trackio Integration

Quantization events are logged to Trackio:

- `quantization_started`: When quantization begins
- `quantization_completed`: When quantization finishes
- `quantized_model_pushed`: When model is uploaded to HF Hub
- `quantization_failed`: If quantization fails

### Metrics Tracked

- Quantization type and parameters
- Model size reduction
- Upload URLs for quantized models
- Processing time and success status

## Dependencies

### Required Packages

```bash
pip install torchao>=0.10.0
pip install transformers>=4.35.0
pip install huggingface_hub>=0.16.0
```

### Optional Dependencies

```bash
pip install accelerate>=0.20.0  # For device mapping
pip install bitsandbytes>=0.41.0  # For additional quantization
```

## References

- [torchao Documentation](https://huggingface.co/docs/transformers/main/en/quantization/torchao)
- [Hugging Face Model Cards](https://huggingface.co/docs/hub/model-cards)
- [Transformers Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization)

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the test files in `tests/test_quantization.py`
3. Open an issue on the project repository
4. Check the Trackio monitoring for detailed logs 