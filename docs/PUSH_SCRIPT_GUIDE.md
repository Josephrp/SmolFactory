# 🚀 Push to Hugging Face Script Guide

## Overview

The `push_to_huggingface.py` script has been enhanced to integrate with **HF Datasets** for experiment tracking and provides complete model deployment with persistent experiment storage.

## 🚀 Key Improvements

### **1. HF Datasets Integration**
- ✅ **Dataset Repository Support**: Configurable dataset repository for experiment storage
- ✅ **Environment Variables**: Automatic detection of `HF_TOKEN` and `TRACKIO_DATASET_REPO`
- ✅ **Enhanced Logging**: Logs push actions to both Trackio and HF Datasets
- ✅ **Model Card Integration**: Includes dataset repository information in model cards

### **2. Enhanced Configuration**
- ✅ **Flexible Token Input**: Multiple ways to provide HF token
- ✅ **Dataset Repository Tracking**: Links models to their experiment datasets
- ✅ **Environment Variable Support**: Fallback to environment variables
- ✅ **Command Line Arguments**: New arguments for HF Datasets integration

### **3. Improved Model Cards**
- ✅ **Dataset Repository Info**: Shows which dataset contains experiment data
- ✅ **Experiment Tracking Section**: Explains how to access training data
- ✅ **Enhanced Documentation**: Better model cards with experiment links

## 📋 Usage Examples

### **Basic Usage**
```bash
# Push model with default settings
python push_to_huggingface.py /path/to/model username/repo-name
```

### **With HF Datasets Integration**
```bash
# Push model with custom dataset repository
python push_to_huggingface.py /path/to/model username/repo-name \
  --dataset-repo username/experiments
```

### **With Custom Token**
```bash
# Push model with custom HF token
python push_to_huggingface.py /path/to/model username/repo-name \
  --hf-token your_token_here
```

### **Complete Example**
```bash
# Push model with all options
python push_to_huggingface.py /path/to/model username/repo-name \
  --dataset-repo username/experiments \
  --hf-token your_token_here \
  --private \
  --experiment-name "smollm3_finetune_v2"
```

## 🔧 Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `model_path` | ✅ Yes | None | Path to trained model directory |
| `repo_name` | ✅ Yes | None | HF repository name (username/repo-name) |
| `--token` | ❌ No | `HF_TOKEN` env | Hugging Face token |
| `--hf-token` | ❌ No | `HF_TOKEN` env | HF token (alternative to --token) |
| `--private` | ❌ No | False | Make repository private |
| `--trackio-url` | ❌ No | None | Trackio Space URL for logging |
| `--experiment-name` | ❌ No | None | Experiment name for Trackio |
| `--dataset-repo` | ❌ No | `TRACKIO_DATASET_REPO` env | HF Dataset repository |

## 🛠️ Configuration Methods

### **Method 1: Command Line Arguments**
```bash
python push_to_huggingface.py model_path repo_name \
  --dataset-repo username/experiments \
  --hf-token your_token_here
```

### **Method 2: Environment Variables**
```bash
export HF_TOKEN=your_token_here
export TRACKIO_DATASET_REPO=username/experiments
python push_to_huggingface.py model_path repo_name
```

### **Method 3: Hybrid Approach**
```bash
# Set defaults via environment variables
export HF_TOKEN=your_token_here
export TRACKIO_DATASET_REPO=username/experiments

# Override specific values via command line
python push_to_huggingface.py model_path repo_name \
  --dataset-repo username/specific-experiments
```

## 📊 What Gets Pushed

### **Model Files**
- ✅ **Model Weights**: `pytorch_model.bin`
- ✅ **Configuration**: `config.json`
- ✅ **Tokenizer**: `tokenizer.json`, `tokenizer_config.json`
- ✅ **All Other Files**: Any additional files in model directory

### **Documentation**
- ✅ **Model Card**: Comprehensive README.md with model information
- ✅ **Training Configuration**: JSON configuration used for training
- ✅ **Training Results**: JSON results and metrics
- ✅ **Training Logs**: Text logs from training process

### **Experiment Data**
- ✅ **Dataset Repository**: Links to HF Dataset containing experiment data
- ✅ **Training Metrics**: All training metrics stored in dataset
- ✅ **Configuration**: Training configuration stored in dataset
- ✅ **Artifacts**: Training artifacts and logs

## 🔍 Enhanced Model Cards

The improved script creates enhanced model cards that include:

### **Model Information**
- Base model and architecture
- Training date and model size
- **Dataset repository** for experiment data

### **Training Configuration**
- Complete training parameters
- Hardware information
- Training duration and steps

### **Experiment Tracking**
- Links to HF Dataset repository
- Instructions for accessing experiment data
- Training metrics and results

### **Usage Examples**
- Code examples for loading and using the model
- Generation examples
- Performance information

## 📈 Logging Integration

### **Trackio Logging**
- ✅ **Push Actions**: Logs model push events
- ✅ **Model Information**: Repository name, size, configuration
- ✅ **Training Data**: Links to experiment dataset

### **HF Datasets Logging**
- ✅ **Experiment Summary**: Final training summary
- ✅ **Push Metadata**: Model repository and push date
- ✅ **Configuration**: Complete training configuration

### **Dual Storage**
- ✅ **Trackio**: Real-time monitoring and visualization
- ✅ **HF Datasets**: Persistent experiment storage
- ✅ **Synchronized**: Both systems updated together

## 🚨 Troubleshooting

### **Issue: "Missing required files"**
**Solutions**:
1. Check model directory contains required files
2. Ensure model was saved correctly during training
3. Verify file permissions

### **Issue: "Failed to create repository"**
**Solutions**:
1. Check HF token has write permissions
2. Verify repository name format: `username/repo-name`
3. Ensure repository doesn't already exist (or use `--private`)

### **Issue: "Failed to upload files"**
**Solutions**:
1. Check network connectivity
2. Verify HF token is valid
3. Ensure repository was created successfully

### **Issue: "Dataset repository not found"**
**Solutions**:
1. Check dataset repository exists
2. Verify HF token has read access
3. Use `--dataset-repo` to specify correct repository

## 📋 Workflow Integration

### **Complete Training Workflow**
1. **Train Model**: Use training scripts with monitoring
2. **Monitor Progress**: View metrics in Trackio interface
3. **Push Model**: Use improved push script
4. **Access Data**: View experiments in HF Dataset repository

### **Example Workflow**
```bash
# 1. Train model with monitoring
python train.py config/train_smollm3_openhermes_fr.py \
  --experiment_name "smollm3_french_v2"

# 2. Push model to HF Hub
python push_to_huggingface.py outputs/model username/smollm3-french \
  --dataset-repo username/experiments \
  --experiment-name "smollm3_french_v2"

# 3. View results
# - Model: https://huggingface.co/username/smollm3-french
# - Experiments: https://huggingface.co/datasets/username/experiments
# - Trackio: Your Trackio Space interface
```

## 🎯 Benefits

### **For Model Deployment**
- ✅ **Complete Documentation**: Enhanced model cards with experiment links
- ✅ **Persistent Storage**: Experiment data stored in HF Datasets
- ✅ **Easy Access**: Direct links to training data and metrics
- ✅ **Reproducibility**: Complete training configuration included

### **For Experiment Management**
- ✅ **Centralized Storage**: All experiments in HF Dataset repository
- ✅ **Version Control**: Model versions linked to experiment data
- ✅ **Collaboration**: Share experiments and models easily
- ✅ **Searchability**: Easy to find specific experiments

### **For Development**
- ✅ **Flexible Configuration**: Multiple ways to set parameters
- ✅ **Backward Compatible**: Works with existing setups
- ✅ **Error Handling**: Clear error messages and troubleshooting
- ✅ **Integration**: Works with existing monitoring system

## 📊 Testing Results

All push script tests passed:
- ✅ **HuggingFacePusher Initialization**: Works with new parameters
- ✅ **Model Card Creation**: Includes HF Datasets integration
- ✅ **Logging Integration**: Logs to both Trackio and HF Datasets
- ✅ **Argument Parsing**: Handles new command line arguments
- ✅ **Environment Variables**: Proper fallback handling

## 🔄 Migration Guide

### **From Old Script**
```bash
# Old way
python push_to_huggingface.py model_path repo_name --token your_token

# New way (same functionality)
python push_to_huggingface.py model_path repo_name --hf-token your_token

# New way with HF Datasets
python push_to_huggingface.py model_path repo_name \
  --hf-token your_token \
  --dataset-repo username/experiments
```

### **Environment Variables**
```bash
# Set environment variables for automatic detection
export HF_TOKEN=your_token_here
export TRACKIO_DATASET_REPO=username/experiments

# Then use simple command
python push_to_huggingface.py model_path repo_name
```

---

**🎉 Your push script is now fully integrated with HF Datasets for complete experiment tracking and model deployment!** 