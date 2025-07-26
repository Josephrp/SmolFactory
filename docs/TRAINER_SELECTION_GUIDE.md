# Trainer Selection Guide

## Overview

This guide explains how to use the new trainer selection feature that allows you to choose between **SFT (Supervised Fine-tuning)** and **DPO (Direct Preference Optimization)** trainers in the SmolLM3 fine-tuning pipeline.

## Trainer Types

### SFT (Supervised Fine-tuning)
- **Purpose**: Standard instruction tuning for most fine-tuning tasks
- **Use Case**: General instruction following, conversation, and task-specific training
- **Dataset Format**: Standard prompt-completion pairs
- **Trainer**: `SmolLM3Trainer` with `SFTTrainer` backend
- **Default**: Yes (default trainer type)

### DPO (Direct Preference Optimization)
- **Purpose**: Preference-based training using human feedback
- **Use Case**: Aligning models with human preferences, reducing harmful outputs
- **Dataset Format**: Preference pairs (chosen/rejected responses)
- **Trainer**: `SmolLM3DPOTrainer` with `DPOTrainer` backend
- **Default**: No (must be explicitly selected)

## Implementation Details

### Configuration Changes

#### Base Config (`config/train_smollm3.py`)
```python
@dataclass
class SmolLM3Config:
    # Trainer type selection
    trainer_type: str = "sft"  # "sft" or "dpo"
    # ... other fields
```

#### DPO Config (`config/train_smollm3_dpo.py`)
```python
@dataclass
class SmolLM3DPOConfig(SmolLM3Config):
    # Trainer type selection
    trainer_type: str = "dpo"  # Override default to use DPO trainer
    # ... DPO-specific fields
```

### Training Script Changes

#### Command Line Arguments
Both `src/train.py` and `scripts/training/train.py` now support:
```bash
--trainer_type {sft,dpo}
```

#### Trainer Selection Logic
```python
# Determine trainer type (command line overrides config)
trainer_type = args.trainer_type or getattr(config, 'trainer_type', 'sft')

# Initialize trainer based on type
if trainer_type.lower() == 'dpo':
    trainer = SmolLM3DPOTrainer(...)
else:
    trainer = SmolLM3Trainer(...)
```

### Launch Script Changes

#### Interactive Selection
The `launch.sh` script now prompts users to select the trainer type:
```
Step 3.5: Trainer Type Selection
====================================

Select the type of training to perform:
1. SFT (Supervised Fine-tuning) - Standard instruction tuning
   - Uses SFTTrainer for instruction following
   - Suitable for most fine-tuning tasks
   - Optimized for instruction datasets

2. DPO (Direct Preference Optimization) - Preference-based training
   - Uses DPOTrainer for preference learning
   - Requires preference datasets (chosen/rejected pairs)
   - Optimizes for human preferences
```

#### Configuration Generation
The generated config file includes the trainer type:
```python
config = SmolLM3Config(
    # Trainer type selection
    trainer_type="$TRAINER_TYPE",
    # ... other fields
)
```

## Usage Examples

### Using the Launch Script
```bash
./launch.sh
# Follow the interactive prompts
# Select "SFT" or "DPO" when prompted
```

### Using Command Line Arguments
```bash
# SFT training (default)
python src/train.py config/train_smollm3.py

# DPO training
python src/train.py config/train_smollm3_dpo.py

# Override trainer type
python src/train.py config/train_smollm3.py --trainer_type dpo
```

### Using the Training Script
```bash
# SFT training
python scripts/training/train.py --config config/train_smollm3.py

# DPO training
python scripts/training/train.py --config config/train_smollm3_dpo.py

# Override trainer type
python scripts/training/train.py --config config/train_smollm3.py --trainer-type dpo
```

## Dataset Requirements

### SFT Training
- **Format**: Standard instruction datasets
- **Fields**: `prompt` and `completion` (or similar)
- **Examples**: OpenHermes, Alpaca, instruction datasets

### DPO Training
- **Format**: Preference datasets
- **Fields**: `chosen` and `rejected` responses
- **Examples**: Human preference datasets, RLHF datasets

## Configuration Priority

1. **Command line argument** (`--trainer_type`) - Highest priority
2. **Config file** (`trainer_type` field) - Medium priority
3. **Default value** (`"sft"`) - Lowest priority

## Monitoring and Logging

Both trainer types support:
- Trackio experiment tracking
- Training metrics logging
- Model checkpointing
- Progress monitoring

## Testing

Run the trainer selection tests:
```bash
python tests/test_trainer_selection.py
```

This verifies:
- Config inheritance works correctly
- Trainer classes exist and are importable
- Trainer type defaults are set correctly

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install trl>=0.7.0 transformers>=4.30.0
   ```

2. **Dataset Format**: DPO requires preference datasets with `chosen`/`rejected` fields

3. **Memory Issues**: DPO training may require more memory due to reference model

4. **Config Conflicts**: Command line arguments override config file settings

### Debugging

Enable verbose logging to see trainer selection:
```bash
python src/train.py config/train_smollm3.py --trainer_type dpo
```

Look for these log messages:
```
Using trainer type: dpo
Initializing DPO trainer...
```

## Future Enhancements

- Support for additional trainer types (RLHF, PPO, etc.)
- Automatic dataset format detection
- Enhanced preference dataset validation
- Multi-objective training support

## Related Documentation

- [Training Configuration Guide](TRAINING_CONFIGURATION_GUIDE.md)
- [Dataset Preparation Guide](DATASET_PREPARATION_GUIDE.md)
- [Monitoring Integration Guide](MONITORING_INTEGRATION_GUIDE.md) 