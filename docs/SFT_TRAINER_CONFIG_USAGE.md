# SFT Trainer Configuration Usage Guide

## Overview

This guide describes how the SFT (Supervised Fine-tuning) trainer uses the premade configuration files and how the `trainer_type` field is passed through the system.

## How SFT Trainer Uses Premade Configs

### 1. Configuration Loading Process

The SFT trainer uses premade configs through the following process:

1. **Config File Selection**: Users specify a config file via command line or launch script
2. **Config Loading**: The system loads the config using `get_config()` function
3. **Config Inheritance**: All configs inherit from `SmolLM3Config` base class
4. **Trainer Type Detection**: The system checks for `trainer_type` field in the config
5. **Training Arguments Creation**: Config parameters are used to create `TrainingArguments`

### 2. Configuration Parameters Used by SFT Trainer

The SFT trainer uses the following config parameters:

#### Model Configuration
- `model_name`: Model to load (e.g., "HuggingFaceTB/SmolLM3-3B")
- `max_seq_length`: Maximum sequence length for tokenization
- `use_flash_attention`: Whether to use flash attention
- `use_gradient_checkpointing`: Whether to use gradient checkpointing

#### Training Configuration
- `batch_size`: Per-device batch size
- `gradient_accumulation_steps`: Gradient accumulation steps
- `learning_rate`: Learning rate for optimization
- `weight_decay`: Weight decay for optimizer
- `warmup_steps`: Number of warmup steps
- `max_iters`: Maximum training iterations
- `save_steps`: Save checkpoint every N steps
- `eval_steps`: Evaluate every N steps
- `logging_steps`: Log every N steps

#### Optimizer Configuration
- `optimizer`: Optimizer type (e.g., "adamw_torch")
- `beta1`, `beta2`, `eps`: Optimizer parameters

#### Scheduler Configuration
- `scheduler`: Learning rate scheduler type
- `min_lr`: Minimum learning rate

#### Mixed Precision
- `fp16`: Whether to use fp16 precision
- `bf16`: Whether to use bf16 precision

#### Data Configuration
- `dataset_name`: Hugging Face dataset name
- `data_dir`: Local dataset directory
- `train_file`: Training file name
- `validation_file`: Validation file name

#### Monitoring Configuration
- `enable_tracking`: Whether to enable Trackio tracking
- `trackio_url`: Trackio server URL
- `experiment_name`: Experiment name for tracking

### 3. Training Arguments Creation

The SFT trainer creates `TrainingArguments` from config parameters:

```python
def get_training_arguments(self, output_dir: str, **kwargs) -> TrainingArguments:
    training_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": self.config.batch_size,
        "per_device_eval_batch_size": self.config.batch_size,
        "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
        "learning_rate": self.config.learning_rate,
        "weight_decay": self.config.weight_decay,
        "warmup_steps": self.config.warmup_steps,
        "max_steps": self.config.max_iters,
        "save_steps": self.config.save_steps,
        "eval_steps": self.config.eval_steps,
        "logging_steps": self.config.logging_steps,
        "fp16": self.config.fp16,
        "bf16": self.config.bf16,
        # ... additional parameters
    }
    return TrainingArguments(**training_args)
```

### 4. Trainer Selection Logic

The system determines which trainer to use based on the `trainer_type` field:

```python
# Determine trainer type (command line overrides config)
trainer_type = args.trainer_type or getattr(config, 'trainer_type', 'sft')

# Initialize trainer based on type
if trainer_type.lower() == 'dpo':
    trainer = SmolLM3DPOTrainer(...)
else:
    trainer = SmolLM3Trainer(...)  # SFT trainer
```

## Configuration Files Structure

### Base Config (`config/train_smollm3.py`)

```python
@dataclass
class SmolLM3Config:
    # Trainer type selection
    trainer_type: str = "sft"  # "sft" or "dpo"
    
    # Model configuration
    model_name: str = "HuggingFaceTB/SmolLM3-3B"
    max_seq_length: int = 4096
    # ... other fields
```

### DPO Config (`config/train_smollm3_dpo.py`)

```python
@dataclass
class SmolLM3DPOConfig(SmolLM3Config):
    # Trainer type selection
    trainer_type: str = "dpo"  # Override default to use DPO trainer
    
    # DPO-specific configuration
    beta: float = 0.1
    # ... DPO-specific fields
```

### Specialized Configs (e.g., `config/train_smollm3_openhermes_fr_a100_multiple_passes.py`)

```python
@dataclass
class SmolLM3ConfigOpenHermesFRMultiplePasses(SmolLM3Config):
    # Inherits trainer_type = "sft" from base config
    
    # Specialized configuration for multiple passes
    batch_size: int = 6
    gradient_accumulation_steps: int = 20
    learning_rate: float = 3e-6
    max_iters: int = 25000
    # ... other specialized fields
```

## Trainer Type Priority

The trainer type is determined in the following order of priority:

1. **Command line argument** (`--trainer_type`) - Highest priority
2. **Config file** (`trainer_type` field) - Medium priority  
3. **Default value** (`"sft"`) - Lowest priority

## Usage Examples

### Using SFT Trainer with Different Configs

```bash
# Basic SFT training (uses base config)
python src/train.py config/train_smollm3.py

# SFT training with specialized config
python src/train.py config/train_smollm3_openhermes_fr_a100_multiple_passes.py

# SFT training with override
python src/train.py config/train_smollm3.py --trainer_type sft

# DPO training (uses DPO config)
python src/train.py config/train_smollm3_dpo.py

# Override config's trainer type
python src/train.py config/train_smollm3.py --trainer_type dpo
```

### Launch Script Usage

```bash
./launch.sh
# Select "SFT" when prompted for trainer type
# The system will use the appropriate config based on selection
```

## Configuration Inheritance

All specialized configs inherit from `SmolLM3Config` and automatically get:

- `trainer_type = "sft"` (default)
- All base training parameters
- All monitoring configuration
- All data configuration

Specialized configs can override any of these parameters for their specific use case.

## SFT Trainer Features

The SFT trainer provides:

1. **SFTTrainer Backend**: Uses Hugging Face's `SFTTrainer` for instruction tuning
2. **Fallback Support**: Falls back to standard `Trainer` if `SFTTrainer` fails
3. **Config Integration**: Uses all config parameters for training setup
4. **Monitoring**: Integrates with Trackio for experiment tracking
5. **Checkpointing**: Supports model checkpointing and resuming
6. **Mixed Precision**: Supports fp16 and bf16 training

## Troubleshooting

### Common Issues

1. **Missing trainer_type field**: Ensure all configs have the `trainer_type` field
2. **Config inheritance issues**: Check that specialized configs properly inherit from base
3. **Parameter conflicts**: Ensure command line arguments don't conflict with config values

### Debugging

Enable verbose logging to see config usage:

```bash
python src/train.py config/train_smollm3.py --trainer_type sft
```

Look for these log messages:
```
Using trainer type: sft
Initializing SFT trainer...
Creating SFTTrainer with training arguments...
```

## Related Documentation

- [Trainer Selection Guide](TRAINER_SELECTION_GUIDE.md)
- [Training Configuration Guide](TRAINING_CONFIGURATION_GUIDE.md)
- [Monitoring Integration Guide](MONITORING_INTEGRATION_GUIDE.md) 