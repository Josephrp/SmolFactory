---
title: Trackio Tonic
emoji: 🐠
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 5.38.0
app_file: app.py
pinned: true
license: mit
short_description: trackio for training monitoring
---

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
    trackio_url="https://huggingface.co/spaces/Tonic/trackio_test_5",
    enable_tracking=True
)
```

Visit: https://huggingface.co/spaces/Tonic/trackio_test_5 