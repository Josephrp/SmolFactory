# H100 Lightweight Training Configuration Guide

This guide explains the new **H100 Lightweight (Rapid)** training configuration, optimized for rapid fine-tuning on H100 GPUs with a small, carefully selected dataset.

## 🎯 Overview

The H100 Lightweight configuration is designed for:
- **Rapid experimentation** on H100 GPUs
- **Efficient training** with 80K carefully selected samples
- **Quick iteration** for research and development
- **Cost-effective** training sessions

## 🚀 Key Features

### **Optimized for H100**
- **Batch Size**: 16 (larger than A100 configs)
- **Gradient Accumulation**: 4 (reduced for faster updates)
- **Learning Rate**: 8e-6 (slightly higher for rapid convergence)
- **Sequence Length**: 8192 (full context window)

### **Dataset Sampling**
- **Source**: OpenHermes-FR dataset
- **Sample Size**: 80,000 random samples
- **Validation**: 1,000 samples (if available)
- **Reproducibility**: Fixed random seed (42)

### **Training Optimizations**
- **Warmup Steps**: 50 (reduced for rapid training)
- **Evaluation**: Every 50 steps
- **Logging**: Every 5 steps
- **Saving**: Every 200 steps
- **Checkpoints**: Keep only 2 (save storage)

## 📊 Configuration Details

### **Model Configuration**
```python
model_name="HuggingFaceTB/SmolLM3-3B"
max_seq_length=8192
use_flash_attention=True
use_gradient_checkpointing=True
```

### **Training Parameters**
```python
batch_size=16
gradient_accumulation_steps=4
learning_rate=8e-6
warmup_steps=50
max_epochs=1
```

### **H100-Specific Optimizations**
```python
dataloader_num_workers=4
dataloader_pin_memory=True
gradient_clipping=1.0
group_by_length=True
pad_to_multiple_of=8
```

### **Memory Optimizations**
```python
save_total_limit=2
early_stopping_patience=3
max_grad_norm=1.0
warmup_ratio=0.1
```

## 🔧 Usage

### **Interactive Selection**
```bash
./launch.sh
# Select "H100 Lightweight (Rapid)" when prompted
```

### **Expected Training Time**
- **H100**: ~2-4 hours (depending on hardware)
- **A100**: ~4-6 hours
- **V100**: ~6-8 hours

### **Memory Requirements**
- **GPU Memory**: 40GB+ (H100 recommended)
- **System RAM**: 32GB+
- **Storage**: 50GB+ for dataset and checkpoints

## 📈 Performance Characteristics

### **Training Speed**
- **Steps per Second**: ~2-3 (on H100)
- **Samples per Second**: ~32-48
- **Effective Batch Size**: 64 (16 × 4)

### **Convergence**
- **Expected Loss**: 1.2-1.8 (after 1 epoch)
- **Evaluation Frequency**: Every 50 steps
- **Early Stopping**: After 3 evaluations without improvement

### **Dataset Efficiency**
- **80K samples**: ~1.3% of full OpenHermes-FR
- **Random sampling**: Ensures diversity
- **Fixed seed**: Reproducible results

## 🎯 Use Cases

### **Perfect For**
- **Rapid prototyping** of new ideas
- **Hyperparameter tuning** experiments
- **Model comparison** studies
- **Research validation** before full training
- **Educational purposes** and learning

### **Not Recommended For**
- **Production models** (use Multiple Passes instead)
- **Competition submissions** (use full dataset)
- **Research papers** (use complete training)

## 🔄 Comparison with Other Configurations

| Configuration | Dataset Size | Batch Size | Epochs | Training Time | Use Case |
|---------------|--------------|------------|--------|---------------|----------|
| **Basic Training** | Full SmolTalk | 2 | 3 | 6-8 hours | Learning |
| **H100 Lightweight** | 80K Hermes-FR | 16 | 1 | 2-4 hours | Rapid experiments |
| **A100 Large Scale** | Full Hermes-FR | 8 | 1.3 | 8-12 hours | Serious research |
| **Multiple Passes** | Full Hermes-FR | 6 | 4 | 24-36 hours | Production |

## 🛠️ Customization

### **Modifying Sample Size**
```bash
# In the launch script, you can modify:
DATASET_SAMPLE_SIZE=50000  # For 50K samples
DATASET_SAMPLE_SIZE=100000 # For 100K samples
```

### **Adjusting Training Parameters**
```bash
# Modify in config/train_smollm3_h100_lightweight.py:
batch_size=12              # Smaller batch size
learning_rate=6e-6         # Lower learning rate
warmup_steps=100          # More warmup steps
```

### **Changing Dataset**
```bash
# Modify the dataset name in the configuration:
dataset_name="your-custom-dataset"
```

## 📊 Monitoring and Results

### **Trackio Integration**
- **Real-time metrics**: Loss, learning rate, gradient norm
- **Training curves**: Visual progress tracking
- **Resource usage**: GPU utilization, memory consumption
- **Artifacts**: Model checkpoints, logs

### **Expected Metrics**
- **Training Loss**: Starts ~3.0, ends ~1.5
- **Validation Loss**: Should be close to training loss
- **Learning Rate**: Cosine decay from 8e-6 to 2e-6
- **Gradient Norm**: Should stay below 1.0

### **Success Indicators**
- **Converging loss**: Steady decrease over time
- **Stable gradients**: Consistent gradient norms
- **Good validation**: Validation loss follows training loss
- **No overfitting**: Validation loss doesn't increase

## 🚨 Troubleshooting

### **Common Issues**

#### **Out of Memory (OOM)**
```bash
# Reduce batch size in config:
batch_size=12  # Instead of 16
gradient_accumulation_steps=6  # Instead of 4
```

#### **Slow Training**
```bash
# Check GPU utilization:
nvidia-smi
# Ensure CUDA is properly installed
python -c "import torch; print(torch.cuda.is_available())"
```

#### **Poor Convergence**
```bash
# Try different learning rate:
learning_rate=6e-6  # Instead of 8e-6
# Or increase warmup:
warmup_steps=100   # Instead of 50
```

#### **Dataset Issues**
```bash
# Check dataset loading:
python -c "from datasets import load_dataset; print(len(load_dataset('legmlai/openhermes-fr')['train']))"
```

### **Performance Tips**

1. **Use H100 if available**: Significantly faster than A100
2. **Monitor GPU memory**: Keep utilization below 90%
3. **Check logs regularly**: Look for convergence issues
4. **Save checkpoints**: Don't lose progress
5. **Use early stopping**: Prevent overfitting

## 📋 Example Workflow

### **Complete H100 Lightweight Training**
```bash
# 1. Setup
python setup_launch.py

# 2. Check requirements
python check_requirements.py

# 3. Run interactive pipeline
./launch.sh

# 4. Select configuration
# Choose: "H100 Lightweight (Rapid)"

# 5. Monitor training
# Watch Trackio Space for real-time progress

# 6. Check results
# Model will be pushed to HF Hub
# Summary in training_summary.md
```

### **Expected Output**
```
✅ Dataset prepared: 80000 train samples, 1000 validation samples
📈 Training started with 5000 total steps
⏱️ Estimated time: 2-4 hours
📊 Monitor progress at: https://huggingface.co/spaces/...
```

## 🎉 Benefits

### **Speed**
- **3-4x faster** than full dataset training
- **Rapid iteration** for research
- **Quick validation** of ideas

### **Efficiency**
- **Reduced costs** (less GPU time)
- **Lower storage** requirements
- **Faster experimentation** cycle

### **Quality**
- **Still high quality** results
- **Good for prototyping**
- **Suitable for many use cases**

## 🔮 Future Enhancements

### **Planned Improvements**
- **Adaptive sampling**: Smart dataset selection
- **Multi-GPU support**: Distributed training
- **Advanced monitoring**: More detailed metrics
- **Auto-tuning**: Automatic hyperparameter optimization

### **Extensibility**
- **Custom datasets**: Easy integration
- **Different models**: Support for other architectures
- **Advanced sampling**: Stratified, balanced sampling

---

**Happy Rapid Training on H100! 🚀** 