# 🔧 Improved Monitoring Integration Guide

## Overview

The monitoring system has been enhanced to support **Hugging Face Datasets** for persistent experiment storage, making it ideal for deployment on Hugging Face Spaces and other cloud environments.

## 🚀 Key Improvements

### 1. **HF Datasets Integration**
- ✅ **Persistent Storage**: Experiments are saved to HF Datasets repositories
- ✅ **Environment Variables**: Configurable via `HF_TOKEN` and `TRACKIO_DATASET_REPO`
- ✅ **Fallback Support**: Graceful degradation if HF Datasets unavailable
- ✅ **Automatic Backup**: Local files as backup

### 2. **Enhanced Monitoring Features**
- 📊 **Real-time Metrics**: Training metrics logged to both Trackio and HF Datasets
- 🔧 **System Metrics**: GPU memory, CPU usage, and system performance
- 📈 **Training Summaries**: Comprehensive experiment summaries
- 🛡️ **Error Handling**: Robust error logging and recovery

### 3. **Easy Integration**
- 🔌 **Automatic Setup**: Environment variables automatically detected
- 📝 **Configuration**: Simple setup with environment variables
- 🔄 **Backward Compatible**: Works with existing Trackio setup

## 📋 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ Yes | None | Your Hugging Face token |
| `TRACKIO_DATASET_REPO` | ❌ No | `tonic/trackio-experiments` | Dataset repository |
| `TRACKIO_URL` | ❌ No | None | Trackio server URL |
| `TRACKIO_TOKEN` | ❌ No | None | Trackio authentication token |

## 🛠️ Setup Instructions

### 1. **Get Your HF Token**
```bash
# Go to https://huggingface.co/settings/tokens
# Create a new token with "Write" permissions
# Copy the token
```

### 2. **Set Environment Variables**
```bash
# For HF Spaces, add these to your Space settings:
HF_TOKEN=your_hf_token_here
TRACKIO_DATASET_REPO=your-username/your-dataset-name

# For local development:
export HF_TOKEN=your_hf_token_here
export TRACKIO_DATASET_REPO=your-username/your-dataset-name
```

### 3. **Create Dataset Repository**
```bash
# Run the setup script
python setup_hf_dataset.py

# Or manually create a dataset on HF Hub
# Go to https://huggingface.co/datasets
# Create a new dataset repository
```

### 4. **Test Configuration**
```bash
# Test your setup
python configure_trackio.py

# Test dataset access
python test_hf_datasets.py
```

## 🚀 Usage Examples

### **Basic Training with Monitoring**
```bash
# Train with default monitoring
python train.py config/train_smollm3_openhermes_fr.py

# Train with custom dataset repository
TRACKIO_DATASET_REPO=your-username/smollm3-experiments python train.py config/train_smollm3_openhermes_fr.py
```

### **Advanced Training Configuration**
```bash
# Train with custom experiment name
python train.py config/train_smollm3_openhermes_fr.py \
  --experiment_name "smollm3_french_tuning_v2" \
  --hf_token your_token_here \
  --dataset_repo your-username/french-experiments
```

### **Training Scripts with Monitoring**
```bash
# All training scripts now support monitoring:
python train.py config/train_smollm3_openhermes_fr_a100_balanced.py
python train.py config/train_smollm3_openhermes_fr_a100_large.py
python train.py config/train_smollm3_openhermes_fr_a100_max_performance.py
python train.py config/train_smollm3_openhermes_fr_a100_multiple_passes.py
```

## 📊 What Gets Monitored

### **Training Metrics**
- Loss values (training and validation)
- Learning rate
- Gradient norms
- Training steps and epochs

### **System Metrics**
- GPU memory usage
- GPU utilization
- CPU usage
- Memory usage

### **Experiment Data**
- Configuration parameters
- Model checkpoints
- Evaluation results
- Training summaries

### **Artifacts**
- Configuration files
- Training logs
- Evaluation results
- Model checkpoints

## 🔍 Viewing Results

### **1. Trackio Interface**
- Visit your Trackio Space
- Navigate to "Experiments" tab
- View real-time metrics and plots

### **2. HF Dataset Repository**
- Go to your dataset repository on HF Hub
- Browse experiment data
- Download experiment files

### **3. Local Files**
- Check local backup files
- Review training logs
- Examine configuration files

## 🛠️ Configuration Examples

### **Default Setup**
```python
# Uses default dataset: tonic/trackio-experiments
# Requires only HF_TOKEN
```

### **Personal Dataset**
```bash
export HF_TOKEN=your_token_here
export TRACKIO_DATASET_REPO=your-username/trackio-experiments
```

### **Team Dataset**
```bash
export HF_TOKEN=your_token_here
export TRACKIO_DATASET_REPO=your-org/team-experiments
```

### **Project-Specific Dataset**
```bash
export HF_TOKEN=your_token_here
export TRACKIO_DATASET_REPO=your-username/smollm3-experiments
```

## 🔧 Troubleshooting

### **Issue: "HF_TOKEN not found"**
```bash
# Solution: Set your HF token
export HF_TOKEN=your_token_here
# Or add to HF Space environment variables
```

### **Issue: "Failed to load dataset"**
```bash
# Solutions:
# 1. Check token has read access
# 2. Verify dataset repository exists
# 3. Run setup script: python setup_hf_dataset.py
```

### **Issue: "Failed to save experiments"**
```bash
# Solutions:
# 1. Check token has write permissions
# 2. Verify dataset repository exists
# 3. Check network connectivity
```

### **Issue: "Monitoring not working"**
```bash
# Solutions:
# 1. Check environment variables
# 2. Run configuration test: python configure_trackio.py
# 3. Check logs for specific errors
```

## 📈 Benefits

### **For HF Spaces Deployment**
- ✅ **Persistent Storage**: Data survives Space restarts
- ✅ **No Local Storage**: No dependency on ephemeral storage
- ✅ **Scalable**: Works with any dataset size
- ✅ **Secure**: Private dataset storage

### **For Experiment Management**
- ✅ **Centralized**: All experiments in one place
- ✅ **Searchable**: Easy to find specific experiments
- ✅ **Versioned**: Dataset versioning for experiments
- ✅ **Collaborative**: Share experiments with team

### **For Development**
- ✅ **Flexible**: Easy to switch between datasets
- ✅ **Configurable**: Environment-based configuration
- ✅ **Robust**: Fallback mechanisms
- ✅ **Debuggable**: Comprehensive logging

## 🎯 Next Steps

1. **Set up your HF token and dataset repository**
2. **Test the configuration with `python configure_trackio.py`**
3. **Run a training experiment to verify monitoring**
4. **Check your HF Dataset repository for experiment data**
5. **View results in your Trackio interface**

## 📚 Related Files

- `monitoring.py` - Enhanced monitoring with HF Datasets support
- `train.py` - Updated training script with monitoring integration
- `configure_trackio.py` - Configuration and testing script
- `setup_hf_dataset.py` - Dataset repository setup
- `test_hf_datasets.py` - Dataset access testing
- `ENVIRONMENT_VARIABLES.md` - Environment variable reference
- `HF_DATASETS_GUIDE.md` - Detailed HF Datasets guide

---

**🎉 Your experiments are now persistently stored and easily accessible!** 