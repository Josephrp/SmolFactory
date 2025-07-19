# A100 Large Scale Training Guide

This guide provides configurations and instructions for running fully-fledged experiments with multiple passes on the full OpenHermes-FR dataset (800k+ datapoints) using A100 GPUs.

## Available Configurations

### 1. A100 Large Batch Configuration
**File**: `config/train_smollm3_openhermes_fr_a100_large.py`

**Key Features**:
- **Effective Batch Size**: 128 (8 × 16 gradient accumulation)
- **Training Duration**: ~1.3 passes (8,000 steps)
- **Learning Rate**: 5e-6 (optimized for large batches)
- **Mixed Precision**: bf16 (A100 optimized)
- **Sequence Length**: 8192 tokens
- **Memory Optimizations**: No gradient checkpointing for A100 efficiency

**Estimated Training Time**: ~6-8 hours on A100

### 2. Multiple Passes Configuration
**File**: `config/train_smollm3_openhermes_fr_a100_multiple_passes.py`

**Key Features**:
- **Effective Batch Size**: 120 (6 × 20 gradient accumulation)
- **Training Duration**: ~4 passes (25,000 steps)
- **Learning Rate**: 3e-6 (conservative for long training)
- **Warmup Steps**: 2000 (longer warmup for stability)
- **Checkpoint Strategy**: More frequent saves (every 2000 steps)

**Estimated Training Time**: ~20-24 hours on A100

## Training Commands

### Quick Start - Large Batch Experiment
```bash
python run_a100_large_experiment.py \
    --config config/train_smollm3_openhermes_fr_a100_large.py \
    --experiment-name "smollm3_openhermes_fr_large_batch" \
    --output-dir ./outputs/large_batch
```

### Multiple Passes Experiment
```bash
python run_a100_large_experiment.py \
    --config config/train_smollm3_openhermes_fr_a100_multiple_passes.py \
    --experiment-name "smollm3_openhermes_fr_multiple_passes" \
    --output-dir ./outputs/multiple_passes
```

### Dry Run (Check Configuration)
```bash
python run_a100_large_experiment.py \
    --config config/train_smollm3_openhermes_fr_a100_large.py \
    --dry-run
```

### Resume Training
```bash
python run_a100_large_experiment.py \
    --config config/train_smollm3_openhermes_fr_a100_multiple_passes.py \
    --resume ./outputs/multiple_passes/checkpoint-10000 \
    --output-dir ./outputs/multiple_passes
```

## Configuration Details

### Memory Usage Optimization
- **Gradient Checkpointing**: Disabled for A100 efficiency
- **Flash Attention**: Enabled for memory efficiency
- **bf16 Mixed Precision**: Better for A100 than fp16
- **Gradient Clipping**: 1.0 for stability
- **Group by Length**: Enabled for better batching

### Data Loading Optimization
- **Num Workers**: 8 for faster data loading
- **Pin Memory**: Enabled for GPU transfer efficiency
- **Prefetch Factor**: 2 for pipeline optimization

### Training Stability
- **Conservative Learning Rate**: Lower LR for large effective batch sizes
- **Longer Warmup**: More warmup steps for stability
- **Higher Beta2**: 0.999 for AdamW stability
- **Gradient Clipping**: Prevents gradient explosion

## Expected Results

### Large Batch Configuration (1.3 passes)
- **Training Steps**: 8,000
- **Effective Batch Size**: 128
- **Steps per Epoch**: ~6,250
- **Epochs**: ~1.3
- **Expected Loss**: Should converge to ~1.5-2.0

### Multiple Passes Configuration (4 passes)
- **Training Steps**: 25,000
- **Effective Batch Size**: 120
- **Steps per Epoch**: ~6,667
- **Epochs**: ~3.75
- **Expected Loss**: Should converge to ~1.2-1.5

## Monitoring and Logging

### Trackio Integration
Both configurations include Trackio monitoring:
- **Metrics Logging**: Every 25-50 steps
- **Artifact Logging**: Model checkpoints
- **Config Logging**: Training configuration

### Checkpoint Strategy
- **Large Batch**: Save every 1000 steps (8 checkpoints)
- **Multiple Passes**: Save every 2000 steps (12 checkpoints)
- **Best Model**: Automatically load best model at end

## Hardware Requirements

### Minimum Requirements
- **GPU**: A100 80GB (or multiple A100s)
- **RAM**: 64GB+ system RAM
- **Storage**: 100GB+ for checkpoints and logs
- **Network**: Fast internet for dataset download

### Recommended Setup
- **GPU**: 2-4x A100 80GB
- **RAM**: 128GB+ system RAM
- **Storage**: 500GB+ NVMe SSD
- **Network**: 10Gbps+ connection

## Troubleshooting

### Out of Memory (OOM)
If you encounter OOM errors:
1. Reduce `batch_size` from 8 to 6 or 4
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_seq_length` from 8192 to 4096

### Slow Training
If training is too slow:
1. Increase `dataloader_num_workers` to 12-16
2. Ensure you're using bf16 mixed precision
3. Check that gradient checkpointing is disabled
4. Verify flash attention is enabled

### Convergence Issues
If loss doesn't converge:
1. Reduce learning rate by 2x
2. Increase warmup steps
3. Check gradient norms in logs
4. Verify dataset quality

## Customization

### For Different Dataset Sizes
Adjust `max_iters` based on your dataset size:
```python
# For 1M datapoints with effective batch size 120
steps_per_epoch = 1000000 // 120  # ~8,333 steps
max_iters = steps_per_epoch * desired_epochs
```

### For Different GPU Memory
Adjust batch size and gradient accumulation:
```python
# For 40GB A100
batch_size = 4
gradient_accumulation_steps = 32  # Effective batch size = 128

# For 24GB GPU
batch_size = 2
gradient_accumulation_steps = 64  # Effective batch size = 128
```

## Performance Tips

1. **Use bf16**: Better than fp16 for A100
2. **Disable Gradient Checkpointing**: A100 has enough memory
3. **Use Flash Attention**: Memory efficient attention
4. **Group by Length**: Better batching efficiency
5. **Pin Memory**: Faster GPU transfers
6. **Multiple Workers**: Faster data loading

## Expected Timeline

- **Large Batch**: 6-8 hours for 1.3 passes
- **Multiple Passes**: 20-24 hours for 4 passes
- **Full Dataset (5+ passes)**: 30+ hours

## Next Steps

After training completes:
1. Evaluate on validation set
2. Test generation quality
3. Push to Hugging Face Hub
4. Deploy for inference

For deployment instructions, see `DEPLOYMENT_GUIDE.md`. 