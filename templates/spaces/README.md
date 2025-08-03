---
title: Track Tonic
emoji: 🐠
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 5.38.0
app_file: app.py
pinned: true
license: mit
short_description: trackio for training monitoring
tags:
- smollm3
- fine-tuned
- causal-lm
- text-generation
- track tonic
- tonic
- legml
---

# Trackio Experiment Tracking

A comprehensive Gradio interface for experiment tracking and monitoring, designed for ML training workflows.

## Features

- **Create Experiments**: Start new experiments with custom names and descriptions
- **Log Metrics**: Real-time logging of training metrics and parameters
- **Visualize Results**: Interactive plots and charts for experiment analysis
- **Manage Status**: Update experiment status (running, completed, failed, paused)
- **HF Datasets Integration**: Persistent storage using Hugging Face Datasets
- **API Access**: Programmatic access for automated training scripts

## Usage

### Web Interface

1. **Create Experiment**: Use the "Create Experiment" tab to start new experiments
2. **Log Metrics**: Use the "Log Metrics" tab to track training progress
3. **View Results**: Use the "View Experiments" tab to see experiment details
4. **Update Status**: Use the "Update Status" tab to mark experiments as completed

### API Integration

Connect your training script to this Trackio Space:

```python
from monitoring import SmolLM3Monitor

monitor = SmolLM3Monitor(
    experiment_name="my_experiment",
    trackio_url="{SPACE_URL}",
    enable_tracking=True
)

# Log configuration
monitor.log_config(config_dict)

# Log metrics during training
monitor.log_metrics({"loss": 0.5, "accuracy": 0.85}, step=100)

# Log final results
monitor.log_training_summary(final_results)
```

## Configuration

### Environment Variables

Set these environment variables for full functionality:

```bash
export HF_TOKEN="your_huggingface_token"
export TRACKIO_DATASET_REPO="your-username/your-dataset"
```

### Dataset Repository

The Space uses Hugging Face Datasets for persistent storage. Create a dataset repository to store your experiments:

1. Go to https://huggingface.co/datasets
2. Create a new dataset repository
3. Set the `TRACKIO_DATASET_REPO` environment variable

## API Endpoints

The Space provides these API endpoints:

- `create_experiment_interface`: Create new experiments
- `log_metrics_interface`: Log training metrics
- `log_parameters_interface`: Log experiment parameters
- `get_experiment_details`: Retrieve experiment details
- `list_experiments_interface`: List all experiments
- `update_experiment_status_interface`: Update experiment status

## Examples

### Creating an Experiment

```python
import requests

response = requests.post(
    "https://your-space.hf.space/gradio_api/call/create_experiment_interface",
    json={"data": ["my_experiment", "Training experiment description"]}
)
```

### Logging Metrics

```python
import requests
import json

metrics = {"loss": 0.5, "accuracy": 0.85, "learning_rate": 2e-5}
response = requests.post(
    "https://your-space.hf.space/gradio_api/call/log_metrics_interface",
    json={"data": ["exp_20231201_143022", json.dumps(metrics), "100"]}
)
```

## Troubleshooting

### Common Issues

1. **Space Not Building**: Check that all required files are uploaded and the app.py is correct
2. **API Connection Errors**: Verify the Space URL and ensure the Space is running
3. **Missing Metrics**: Check that the experiment ID is correct and the Space is accessible

### Getting Help

- Check the Space logs at the Space URL
- Verify your HF token has the necessary permissions
- Ensure the dataset repository exists and is accessible

Visit: {SPACE_URL} 