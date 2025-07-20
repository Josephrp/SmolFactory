# 🚀 Trackio with Hugging Face Datasets - Complete Guide

## Overview

This guide explains how to use Hugging Face Datasets for persistent storage of Trackio experiments, providing reliable data persistence across Hugging Face Spaces deployments.

## 🏗️ Architecture

### Why HF Datasets?

1. **Persistent Storage**: Data survives Space restarts and redeployments
2. **Version Control**: Automatic versioning of experiment data
3. **Access Control**: Private datasets for security
4. **Reliability**: HF's infrastructure ensures data availability
5. **Scalability**: Handles large amounts of experiment data

### Data Flow

```
Training Script → Trackio App → HF Dataset → Trackio App → Plots
```

## 🚀 Setup Instructions

### 1. Create HF Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with `write` permissions
3. Copy the token for use in your Space

### 2. Set Up Dataset Repository

```bash
# Run the setup script
python setup_hf_dataset.py
```

This will:
- Create a private dataset: `tonic/trackio-experiments`
- Add your existing experiments
- Configure the dataset for Trackio

### 3. Configure Hugging Face Space

#### Environment Variables
Set these in your HF Space settings:
```bash
HF_TOKEN=your_hf_token_here
TRACKIO_DATASET_REPO=your-username/your-dataset-name
```

**Environment Variables Explained:**
- `HF_TOKEN`: Your Hugging Face token (required for dataset access)
- `TRACKIO_DATASET_REPO`: Dataset repository to use (optional, defaults to `tonic/trackio-experiments`)

**Example Configurations:**
```bash
# Use default dataset
HF_TOKEN=your_token_here

# Use personal dataset
HF_TOKEN=your_token_here
TRACKIO_DATASET_REPO=your-username/trackio-experiments

# Use team dataset
HF_TOKEN=your_token_here
TRACKIO_DATASET_REPO=your-org/team-experiments

# Use project-specific dataset
HF_TOKEN=your_token_here
TRACKIO_DATASET_REPO=your-username/smollm3-experiments
```

#### Requirements
Update your `requirements.txt`:
```txt
gradio>=4.0.0
plotly>=5.0.0
pandas>=1.5.0
numpy>=1.24.0
datasets>=2.14.0
huggingface-hub>=0.16.0
requests>=2.31.0
```

### 4. Deploy Updated App

The updated `app.py` now:
- Loads experiments from HF Dataset
- Saves new experiments to the dataset
- Falls back to backup data if dataset unavailable
- Provides better error handling

### 5. Configure Environment Variables

Use the configuration script to check your setup:

```bash
python configure_trackio.py
```

This script will:
- Show current environment variables
- Test dataset access
- Generate configuration file
- Provide usage examples

**Available Environment Variables:**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | None | Your Hugging Face token |
| `TRACKIO_DATASET_REPO` | No | `tonic/trackio-experiments` | Dataset repository to use |
| `SPACE_ID` | Auto | None | HF Space ID (auto-detected) |

## 📊 Dataset Schema

The HF Dataset contains these columns:

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | string | Unique experiment identifier |
| `name` | string | Experiment name |
| `description` | string | Experiment description |
| `created_at` | string | ISO timestamp |
| `status` | string | running/completed/failed |
| `metrics` | string | JSON array of metric entries |
| `parameters` | string | JSON object of experiment parameters |
| `artifacts` | string | JSON array of artifacts |
| `logs` | string | JSON array of log entries |
| `last_updated` | string | ISO timestamp of last update |

## 🔧 Technical Details

### Loading Experiments

```python
from datasets import load_dataset

# Load from HF Dataset
dataset = load_dataset("tonic/trackio-experiments", token=HF_TOKEN)

# Convert to experiments dict
for row in dataset['train']:
    experiment = {
        'id': row['experiment_id'],
        'metrics': json.loads(row['metrics']),
        'parameters': json.loads(row['parameters']),
        # ... other fields
    }
```

### Saving Experiments

```python
from datasets import Dataset
from huggingface_hub import HfApi

# Convert experiments to dataset format
dataset_data = []
for exp_id, exp_data in experiments.items():
    dataset_data.append({
        'experiment_id': exp_id,
        'metrics': json.dumps(exp_data['metrics']),
        'parameters': json.dumps(exp_data['parameters']),
        # ... other fields
    })

# Push to HF Hub
dataset = Dataset.from_list(dataset_data)
dataset.push_to_hub("tonic/trackio-experiments", token=HF_TOKEN, private=True)
```

## 📈 Your Current Experiments

### Available Experiments

1. **`exp_20250720_130853`** (petite-elle-l-aime-3)
   - 4 metric entries (steps 25, 50, 75, 100)
   - Loss decreasing: 1.1659 → 1.1528
   - Good convergence pattern

2. **`exp_20250720_134319`** (petite-elle-l-aime-3-1)
   - 2 metric entries (step 25)
   - Loss: 1.166
   - GPU memory tracking

### Metrics Available for Plotting

- `loss` - Training loss curve
- `learning_rate` - Learning rate schedule
- `mean_token_accuracy` - Token-level accuracy
- `grad_norm` - Gradient norm
- `num_tokens` - Tokens processed
- `epoch` - Training epoch
- `gpu_0_memory_allocated` - GPU memory usage
- `cpu_percent` - CPU usage
- `memory_percent` - System memory

## 🎯 Usage Instructions

### 1. View Experiments
- Go to "View Experiments" tab
- Enter experiment ID: `exp_20250720_130853` or `exp_20250720_134319`
- Click "View Experiment"

### 2. Create Plots
- Go to "Visualizations" tab
- Enter experiment ID
- Select metric to plot
- Click "Create Plot"

### 3. Compare Experiments
- Use "Experiment Comparison" feature
- Enter: `exp_20250720_130853,exp_20250720_134319`
- Compare loss curves

## 🔍 Troubleshooting

### Issue: "No metrics data available"
**Solutions**:
1. Check HF_TOKEN is set correctly
2. Verify dataset repository exists
3. Check network connectivity to HF Hub

### Issue: "Failed to load from dataset"
**Solutions**:
1. App falls back to backup data automatically
2. Check dataset permissions
3. Verify token has read access

### Issue: "Failed to save experiments"
**Solutions**:
1. Check token has write permissions
2. Verify dataset repository exists
3. Check network connectivity

## 🚀 Benefits of This Approach

### ✅ Advantages
- **Persistent**: Data survives Space restarts
- **Reliable**: HF's infrastructure ensures availability
- **Secure**: Private datasets protect your data
- **Scalable**: Handles large amounts of experiment data
- **Versioned**: Automatic versioning of experiment data

### 🔄 Fallback Strategy
1. **Primary**: Load from HF Dataset
2. **Secondary**: Use backup data (your existing experiments)
3. **Tertiary**: Create new experiments locally

## 📋 Next Steps

1. **Set HF_TOKEN**: Add your token to Space environment
2. **Run Setup**: Execute `setup_hf_dataset.py`
3. **Deploy App**: Push updated `app.py` to your Space
4. **Test Plots**: Verify experiments load and plots work
5. **Monitor Training**: New experiments will be saved to dataset

## 🔐 Security Notes

- Dataset is **private** by default
- Only accessible with your HF_TOKEN
- Experiment data is stored securely on HF infrastructure
- No sensitive data is exposed publicly

---

**Your experiments are now configured for reliable persistence using Hugging Face Datasets!** 🎉 