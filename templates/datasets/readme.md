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
- track tonic
- tonic
- experiment tracking
- smollm3
- fine-tuning
- legml
- hermes
---

# Trackio Experiments Dataset

This dataset stores experiment tracking data for ML training runs, particularly focused on SmolLM3 fine-tuning experiments with comprehensive metrics tracking.

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

## Metrics Structure

The metrics field contains JSON arrays with the following structure:

```json
[
  {
    "timestamp": "2025-07-20T11:20:01.780908",
    "step": 25,
    "metrics": {
      "loss": 1.1659,
      "accuracy": 0.759,
      "learning_rate": 7e-08,
      "grad_norm": 10.3125,
      "epoch": 0.004851130919895701,
      
      // Advanced Training Metrics
      "total_tokens": 1642080.0,
      "truncated_tokens": 128,
      "padding_tokens": 256,
      "throughput": 3284160.0,
      "step_time": 0.5,
      "batch_size": 8,
      "seq_len": 2048,
      "token_acc": 0.759,
      
      // Custom Losses
      "train/gate_ortho": 0.0234,
      "train/center": 0.0156,
      
      // System Metrics
      "gpu_memory_allocated": 17.202261447906494,
      "gpu_memory_reserved": 75.474609375,
      "gpu_utilization": 85.2,
      "cpu_percent": 2.7,
      "memory_percent": 10.1
    }
  }
]
```

## Supported Metrics

### Core Training Metrics
- **loss**: Training loss value
- **accuracy**: Model accuracy
- **learning_rate**: Current learning rate
- **grad_norm**: Gradient norm
- **epoch**: Current epoch progress

### Advanced Token Metrics
- **total_tokens**: Total tokens processed in the batch
- **truncated_tokens**: Number of tokens truncated during processing
- **padding_tokens**: Number of padding tokens added
- **throughput**: Tokens processed per second
- **step_time**: Time taken for the current training step
- **batch_size**: Current batch size
- **seq_len**: Sequence length
- **token_acc**: Token-level accuracy

### Custom Losses (SmolLM3-specific)
- **train/gate_ortho**: Gate orthogonality loss
- **train/center**: Center loss component

### System Performance Metrics
- **gpu_memory_allocated**: GPU memory currently allocated (GB)
- **gpu_memory_reserved**: GPU memory reserved (GB)
- **gpu_utilization**: GPU utilization percentage
- **cpu_percent**: CPU usage percentage
- **memory_percent**: System memory usage percentage

## Usage

This dataset is automatically used by the Trackio monitoring system to store and retrieve experiment data. It provides persistent storage for experiment tracking across different training runs.

## Integration

The dataset is used by:
- Trackio Spaces for experiment visualization
- Training scripts for logging metrics and parameters
- Monitoring systems for experiment tracking
- SmolLM3 fine-tuning pipeline for comprehensive metrics capture

## Privacy

This dataset is private by default to ensure experiment data security. Only users with appropriate permissions can access the data.

## Examples

### Sample Experiment Entry
```json
{
  "experiment_id": "exp_20250720_130853",
  "name": "smollm3_finetune",
  "description": "SmolLM3 fine-tuning experiment with comprehensive metrics",
  "created_at": "2025-07-20T11:20:01.780908",
  "status": "running",
  "metrics": "[{\"timestamp\": \"2025-07-20T11:20:01.780908\", \"step\": 25, \"metrics\": {\"loss\": 1.1659, \"accuracy\": 0.759, \"total_tokens\": 1642080.0, \"throughput\": 3284160.0, \"train/gate_ortho\": 0.0234, \"train/center\": 0.0156}}]",
  "parameters": "{\"model_name\": \"HuggingFaceTB/SmolLM3-3B\", \"batch_size\": 8, \"learning_rate\": 3.5e-06, \"max_seq_length\": 12288}",
  "artifacts": "[]",
  "logs": "[]",
  "last_updated": "2025-07-20T11:20:01.780908"
}
```

## License

This dataset is part of the Trackio experiment tracking system and follows the same license as the main project.
