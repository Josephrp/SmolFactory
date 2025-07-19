# Trackio Integration for SmolLM3 Fine-tuning

This document provides comprehensive information about the Trackio experiment tracking and monitoring integration for your SmolLM3 fine-tuning pipeline.

## Features

- **SmolLM3 Fine-tuning**: Support for supervised fine-tuning and DPO training
- **Trackio Integration**: Complete experiment tracking and monitoring
- **Hugging Face Spaces Deployment**: Easy deployment of Trackio monitoring interface
- **Comprehensive Logging**: Metrics, parameters, artifacts, and system monitoring
- **Flexible Configuration**: Support for various training configurations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Training with Trackio

```bash
python train.py config/train_smollm3.py \
    --dataset_dir my_dataset \
    --enable_tracking \
    --trackio_url "https://your-trackio-instance.com" \
    --experiment_name "smollm3_finetune_v1"
```

### 3. Training with Custom Parameters

```bash
python train.py config/train_smollm3.py \
    --dataset_dir my_dataset \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --max_iters 2000 \
    --enable_tracking \
    --trackio_url "https://your-trackio-instance.com" \
    --experiment_name "smollm3_high_lr_experiment"
```

## Trackio Integration

### Configuration

Add Trackio settings to your configuration:

```python
# In your config file
config = SmolLM3Config(
    # ... other settings ...
    
    # Trackio monitoring configuration
    enable_tracking=True,
    trackio_url="https://your-trackio-instance.com",
    trackio_token="your_token_here",  # Optional
    log_artifacts=True,
    log_metrics=True,
    log_config=True,
    experiment_name="my_experiment"
)
```

### Environment Variables

You can also set Trackio configuration via environment variables:

```bash
export TRACKIO_URL="https://your-trackio-instance.com"
export TRACKIO_TOKEN="your_token_here"
```

### What Gets Tracked

- **Configuration**: All training parameters and model settings
- **Metrics**: Loss, accuracy, learning rate, and custom metrics
- **System Metrics**: GPU memory, CPU usage, training time
- **Artifacts**: Model checkpoints, evaluation results
- **Training Summary**: Final results and experiment duration

## Hugging Face Spaces Deployment

### Deploy Trackio Monitoring Interface

1. **Create a new Space** on Hugging Face:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Gradio" as the SDK
   - Set visibility (Public or Private)

2. **Upload the deployment files**:
   - `app.py` - The Gradio interface
   - `requirements_space.txt` - Dependencies
   - `README.md` - Documentation

3. **Configure the Space**:
   - The Space will automatically install dependencies
   - The Gradio interface will be available at your Space URL

### Using the Trackio Space

1. **Create Experiments**: Use the "Create Experiment" tab to start new experiments
2. **Log Metrics**: Use the "Log Metrics" tab to track training progress
3. **View Results**: Use the "View Experiments" tab to see experiment details
4. **Update Status**: Use the "Update Status" tab to mark experiments as completed

### Integration with Your Training

To connect your training script to the Trackio Space:

```python
# In your training script
from monitoring import SmolLM3Monitor

# Initialize monitor
monitor = SmolLM3Monitor(
    experiment_name="my_experiment",
    trackio_url="https://your-space.hf.space",  # Your Space URL
    enable_tracking=True
)

# Log configuration
monitor.log_config(config_dict)

# Log metrics during training
monitor.log_metrics({"loss": 0.5, "accuracy": 0.85}, step=100)

# Log final results
monitor.log_training_summary(final_results)
```

## Configuration Files

### Main Configuration (`config/train_smollm3.py`)

```python
@dataclass
class SmolLM3Config:
    # Model configuration
    model_name: str = "HuggingFaceTB/SmolLM3-3B"
    max_seq_length: int = 4096
    
    # Training configuration
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_iters: int = 1000
    
    # Trackio monitoring
    enable_tracking: bool = True
    trackio_url: Optional[str] = None
    trackio_token: Optional[str] = None
    experiment_name: Optional[str] = None
```

### DPO Configuration (`config/train_smollm3_dpo.py`)

```python
@dataclass
class SmolLM3DPOConfig(SmolLM3Config):
    # DPO-specific settings
    beta: float = 0.1
    max_prompt_length: int = 2048
    
    # Trackio monitoring (inherited)
    enable_tracking: bool = True
    trackio_url: Optional[str] = None
```

## Monitoring Features

### Real-time Metrics

- Training loss and evaluation metrics
- Learning rate scheduling
- GPU memory and utilization
- Training time and progress

### Artifact Tracking

- Model checkpoints at regular intervals
- Evaluation results and plots
- Configuration snapshots
- Training logs and summaries

### Experiment Management

- Experiment naming and organization
- Status tracking (running, completed, failed)
- Parameter comparison across experiments
- Result visualization

## Advanced Usage

### Custom Metrics

```python
# Log custom metrics
monitor.log_metrics({
    "custom_metric": value,
    "perplexity": perplexity_score,
    "bleu_score": bleu_score
}, step=current_step)
```

### System Monitoring

```python
# Log system metrics
monitor.log_system_metrics(step=current_step)
```

### Artifact Logging

```python
# Log model checkpoint
monitor.log_model_checkpoint("checkpoint-1000", step=1000)

# Log evaluation results
monitor.log_evaluation_results(eval_results, step=1000)
```

## Troubleshooting

### Common Issues

1. **Trackio not available**: Install with `pip install trackio`
2. **Connection errors**: Check your Trackio URL and token
3. **Missing metrics**: Ensure monitoring is enabled in configuration
4. **Space deployment issues**: Check Gradio version compatibility

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 