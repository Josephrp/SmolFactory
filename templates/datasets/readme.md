---
dataset_info:
  features:
  - name: experiment_id
    dtype: string
  - name: name
    dtype: string
  - name: description
    dtype: string
  - name: created_at
    dtype: string
  - name: status
    dtype: string
  - name: metrics
    dtype: string
  - name: parameters
    dtype: string
  - name: artifacts
    dtype: string
  - name: logs
    dtype: string
  - name: last_updated
    dtype: string
  splits:
  - name: train
    num_bytes: 4945
    num_examples: 2
  download_size: 15529
  dataset_size: 4945
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
tags:
- trackio
- tonic
- experiment tracking
---

# Trackio Experiments Dataset

This dataset stores experiment tracking data for ML training runs, particularly focused on SmolLM3 fine-tuning experiments.

## Dataset Structure

The dataset contains the following columns:

- **experiment_id**: Unique identifier for each experiment
- **name**: Human-readable name for the experiment
- **description**: Detailed description of the experiment
- **created_at**: Timestamp when the experiment was created
- **status**: Current status (running, completed, failed, paused)
- **metrics**: JSON string containing training metrics over time
- **parameters**: JSON string containing experiment configuration
- **artifacts**: JSON string containing experiment artifacts
- **logs**: JSON string containing experiment logs
- **last_updated**: Timestamp of last update

## Usage

This dataset is automatically used by the Trackio monitoring system to store and retrieve experiment data. It provides persistent storage for experiment tracking across different training runs.

## Integration

The dataset is used by:
- Trackio Spaces for experiment visualization
- Training scripts for logging metrics and parameters
- Monitoring systems for experiment tracking

## Privacy

This dataset is private by default to ensure experiment data security. Only users with appropriate permissions can access the data.

## Examples

### Sample Experiment Entry
```json
{
  "experiment_id": "exp_20250720_130853",
  "name": "smollm3_finetune",
  "description": "SmolLM3 fine-tuning experiment",
  "created_at": "2025-07-20T11:20:01.780908",
  "status": "running",
  "metrics": "[{\"timestamp\": \"2025-07-20T11:20:01.780908\", \"step\": 25, \"metrics\": {\"loss\": 1.1659, \"accuracy\": 0.759}}]",
  "parameters": "{\"model_name\": \"HuggingFaceTB/SmolLM3-3B\", \"batch_size\": 8, \"learning_rate\": 3.5e-06}",
  "artifacts": "[]",
  "logs": "[]",
  "last_updated": "2025-07-20T11:20:01.780908"
}
```

## License

This dataset is part of the Trackio experiment tracking system and follows the same license as the main project.
