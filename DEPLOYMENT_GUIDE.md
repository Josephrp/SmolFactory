# Trackio Deployment Guide for Hugging Face Spaces

This guide provides step-by-step instructions for deploying Trackio experiment tracking to Hugging Face Spaces and integrating it with your SmolLM3 fine-tuning pipeline.

## Prerequisites

- Hugging Face account
- Hugging Face CLI installed (`pip install huggingface_hub`)
- Git configured with your Hugging Face credentials

## Method 1: Automated Deployment (Recommended)

### Step 1: Run the Deployment Script

```bash
python deploy_trackio_space.py
```

The script will prompt you for:
- Your Hugging Face username
- Space name (e.g., `trackio-monitoring`)
- Hugging Face token (needs a write token obviously)

### Step 2: Wait for Build

After deployment, wait 2-5 minutes for the Space to build and become available.

### Step 3: Test the Interface

Visit your Space URL to test the interface:
```
https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

## Method 2: Manual Deployment

### Step 1: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Configure the Space:
   - **Owner**: Your username
   - **Space name**: `trackio-monitoring` (or your preferred name)
   - **SDK**: Gradio
   - **Hardware**: CPU (Basic)
   - **License**: MIT

### Step 2: Upload Files

Upload these files to your Space:

#### `app.py`
The main Gradio interface (already created in this repository)

#### `requirements_space.txt`
```
gradio>=4.0.0
gradio-client>=0.10.0
requests>=2.31.0
numpy>=1.24.0
pandas>=2.0.0
jsonschema>=4.17.0
plotly>=5.15.0
matplotlib>=3.7.0
python-dotenv>=1.0.0
```

#### `README.md`
```markdown
# Trackio Experiment Tracking

A Gradio interface for experiment tracking and monitoring.

## Features

- Create and manage experiments
- Log training metrics and parameters
- View experiment details and results
- Update experiment status

## Usage

1. Create a new experiment using the "Create Experiment" tab
2. Log metrics during training using the "Log Metrics" tab
3. View experiment details using the "View Experiments" tab
4. Update experiment status using the "Update Status" tab

## Integration

To connect your training script to this Trackio Space:

```python
from monitoring import SmolLM3Monitor

monitor = SmolLM3Monitor(
    experiment_name="my_experiment",
    trackio_url="https://your-space.hf.space",
    enable_tracking=True
)
```

### Step 3: Configure Space Settings

In your Space settings, ensure:
- **App file**: `app.py`
- **Python version**: 3.9 or higher
- **Hardware**: CPU (Basic) is sufficient

## Integration with Your Training Script

### Step 1: Update Your Configuration

Add Trackio settings to your training configuration:

```python
# config/train_smollm3.py
@dataclass
class SmolLM3Config:
    # ... existing settings ...
    
    # Trackio monitoring configuration
    enable_tracking: bool = True
    trackio_url: Optional[str] = None  # Your Space URL
    trackio_token: Optional[str] = None
    log_artifacts: bool = True
    log_metrics: bool = True
    log_config: bool = True
    experiment_name: Optional[str] = None
```

### Step 2: Run Training with Trackio

```bash
python train.py config/train_smollm3.py \
    --dataset_dir my_dataset \
    --enable_tracking \
    --trackio_url "https://your-username-trackio-monitoring.hf.space" \
    --experiment_name "smollm3_finetune_v1"
```

### Step 3: Monitor Your Experiments

1. **Create Experiment**: Use the "Create Experiment" tab in your Space
2. **Log Metrics**: Your training script will automatically log metrics
3. **View Results**: Use the "View Experiments" tab to see progress
4. **Update Status**: Mark experiments as completed when done

## Advanced Configuration

### Environment Variables

You can set Trackio configuration via environment variables:

```bash
export TRACKIO_URL="https://your-space.hf.space"
export TRACKIO_TOKEN="your_token_here"
```

### Custom Experiment Names

```bash
python train.py config/train_smollm3.py \
    --experiment_name "smollm3_high_lr_experiment" \
    --trackio_url "https://your-space.hf.space"
```

### Multiple Experiments

You can run multiple experiments and track them separately:

```bash
# Experiment 1
python train.py config/train_smollm3.py \
    --experiment_name "smollm3_baseline" \
    --learning_rate 2e-5

# Experiment 2
python train.py config/train_smollm3.py \
    --experiment_name "smollm3_high_lr" \
    --learning_rate 5e-5
```

## Using the Trackio Interface

### Creating Experiments

1. Go to the "Create Experiment" tab
2. Enter experiment name (e.g., "smollm3_finetune_v1")
3. Add description (optional)
4. Click "Create Experiment"
5. Note the experiment ID for logging metrics

### Logging Metrics

1. Go to the "Log Metrics" tab
2. Enter your experiment ID
3. Add metrics in JSON format:
   ```json
   {
     "loss": 0.5,
     "accuracy": 0.85,
     "learning_rate": 2e-5
   }
   ```
4. Add step number (optional)
5. Click "Log Metrics"

### Viewing Experiments

1. Go to the "View Experiments" tab
2. Enter experiment ID to view specific experiment
3. Or click "List All Experiments" to see all experiments

### Updating Status

1. Go to the "Update Status" tab
2. Enter experiment ID
3. Select new status (running, completed, failed, paused)
4. Click "Update Status"

## Troubleshooting

### Common Issues

#### 1. Space Not Building
- Check that all required files are uploaded
- Verify `app.py` is the main file
- Check the Space logs for errors

#### 2. Connection Errors
- Verify your Space URL is correct
- Check that the Space is running (not paused)
- Ensure your training script can reach the Space URL

#### 3. Missing Metrics
- Check that `enable_tracking=True` in your config
- Verify the Trackio URL is correct
- Check training logs for monitoring errors

#### 4. Authentication Issues
- If using tokens, verify they're correct
- Check Hugging Face account permissions
- Ensure Space is public or you have access

### Debug Mode

Enable debug logging in your training script:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Testing

Test the Trackio interface manually:

1. Create an experiment
2. Log some test metrics
3. View the experiment details
4. Update the status

## Security Considerations

### Public vs Private Spaces

- **Public Spaces**: Anyone can view and use the interface
- **Private Spaces**: Only you and collaborators can access

### Token Management

- Store tokens securely (environment variables)
- Don't commit tokens to version control
- Use Hugging Face's token management

### Data Privacy

- Trackio stores experiment data in the Space
- Consider data retention policies
- Be mindful of sensitive information in experiment names

## Performance Optimization

### Space Configuration

- Use CPU (Basic) for the interface (sufficient for tracking)
- Consider GPU only for actual training
- Monitor Space usage and limits

### Efficient Logging

- Log metrics at reasonable intervals (every 10-100 steps)
- Avoid logging too frequently to prevent rate limiting
- Use batch logging when possible

## Monitoring Best Practices

### Experiment Naming

Use descriptive names:
- `smollm3_baseline_v1`
- `smollm3_high_lr_experiment`
- `smollm3_dpo_training`

### Metric Logging

Log relevant metrics:
- Training loss
- Validation loss
- Learning rate
- GPU memory usage
- Training time

### Status Management

- Mark experiments as "running" when starting
- Update to "completed" when finished
- Mark as "failed" if errors occur
- Use "paused" for temporary stops

## Integration Examples

### Basic Integration

```python
from monitoring import SmolLM3Monitor

# Initialize monitor
monitor = SmolLM3Monitor(
    experiment_name="my_experiment",
    trackio_url="https://your-space.hf.space",
    enable_tracking=True
)

# Log configuration
monitor.log_config(config_dict)

# Log metrics during training
monitor.log_metrics({"loss": 0.5}, step=100)

# Log final results
monitor.log_training_summary(final_results)
```

### Advanced Integration

```python
# Custom monitoring setup
monitor = SmolLM3Monitor(
    experiment_name="smollm3_advanced",
    trackio_url="https://your-space.hf.space",
    enable_tracking=True,
    log_artifacts=True,
    log_metrics=True,
    log_config=True
)

# Log system metrics
monitor.log_system_metrics(step=current_step)

# Log model checkpoint
monitor.log_model_checkpoint("checkpoint-1000", step=1000)

# Log evaluation results
monitor.log_evaluation_results(eval_results, step=1000)
```

## Support and Resources

### Documentation

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [Trackio GitHub Repository](https://github.com/Josephrp/trackio)

### Community

- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Gradio Discord](https://discord.gg/feTf9z3Z)

### Issues and Feedback

- Report issues on the project repository
- Provide feedback on the Trackio interface
- Suggest improvements for the monitoring system

## Conclusion

You now have a complete Trackio monitoring system deployed on Hugging Face Spaces! This setup provides:

- ✅ Easy experiment tracking and monitoring
- ✅ Real-time metric logging
- ✅ Web-based interface for experiment management
- ✅ Integration with your SmolLM3 fine-tuning pipeline
- ✅ Scalable and accessible monitoring solution

Start tracking your experiments and gain insights into your model training process! 