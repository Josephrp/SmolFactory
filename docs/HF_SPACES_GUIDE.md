# 🚀 Trackio on Hugging Face Spaces - Complete Guide

## Overview

This guide explains how to properly deploy and use Trackio on Hugging Face Spaces, addressing the unique challenges of ephemeral storage and data persistence.

## 🏗️ Hugging Face Spaces Architecture

### Key Challenges

1. **Ephemeral Storage**: File system gets reset between deployments
2. **No Persistent Storage**: Files written during runtime don't persist
3. **Multiple Instances**: Training and monitoring might run in different environments
4. **Limited File System**: Restricted write permissions in certain directories

### How Trackio Handles HF Spaces

The updated Trackio app now includes:

- **Automatic HF Spaces Detection**: Detects when running on HF Spaces
- **Persistent Path Selection**: Uses `/tmp/` for better persistence
- **Backup Recovery**: Automatically recovers experiments from backup data
- **Fallback Storage**: Multiple storage locations for redundancy

## 📊 Your Current Experiments

Based on your logs, you have these experiments available:

### Experiment 1: `exp_20250720_130853`
- **Name**: petite-elle-l-aime-3
- **Status**: Running
- **Metrics**: 4 entries (steps 25, 50, 75, 100)
- **Key Metrics**: Loss decreasing from 1.1659 to 1.1528

### Experiment 2: `exp_20250720_134319`
- **Name**: petite-elle-l-aime-3-1
- **Status**: Running
- **Metrics**: 2 entries (step 25)
- **Key Metrics**: Loss 1.166, GPU memory usage

## 🎯 How to Use Your Experiments

### 1. View Experiments
- Go to the "View Experiments" tab
- Enter experiment ID: `exp_20250720_130853` or `exp_20250720_134319`
- Click "View Experiment" to see details

### 2. Create Plots
- Go to the "Visualizations" tab
- Enter experiment ID
- Select metric to plot:
  - `loss` - Training loss curve
  - `learning_rate` - Learning rate schedule
  - `mean_token_accuracy` - Token accuracy
  - `grad_norm` - Gradient norm
  - `gpu_0_memory_allocated` - GPU memory usage

### 3. Compare Experiments
- Use the "Experiment Comparison" feature
- Enter: `exp_20250720_130853,exp_20250720_134319`
- Compare loss curves between experiments

## 🔧 Technical Details

### Data Persistence Strategy

```python
# HF Spaces detection
if os.environ.get('SPACE_ID'):
    data_file = "/tmp/trackio_experiments.json"
else:
    data_file = "trackio_experiments.json"
```

### Backup Recovery

The app automatically recovers your experiments from backup data when:
- Running on HF Spaces
- No existing experiments found
- Data file is missing or empty

### Storage Locations

1. **Primary**: `/tmp/trackio_experiments.json`
2. **Backup**: `/tmp/trackio_backup.json`
3. **Fallback**: Local directory (for development)

## 🚀 Deployment Best Practices

### 1. Environment Variables
```bash
# Set in HF Spaces environment
SPACE_ID=your-space-id
TRACKIO_URL=https://your-space.hf.space
```

### 2. File Structure
```
your-space/
├── app.py                 # Main Trackio app
├── requirements.txt       # Dependencies
├── README.md             # Space description
└── .gitignore           # Ignore temporary files
```

### 3. Requirements
```txt
gradio>=4.0.0
plotly>=5.0.0
pandas>=1.5.0
numpy>=1.24.0
```

## 📈 Monitoring Your Training

### Real-time Metrics
Your experiments show:
- **Loss**: Decreasing from 1.1659 to 1.1528 (good convergence)
- **Learning Rate**: Properly scheduled from 7e-08 to 2.8875e-07
- **Token Accuracy**: Around 75-76% (reasonable for early training)
- **GPU Memory**: ~17GB allocated, 75GB reserved

### Expected Behavior
- Loss should continue decreasing
- Learning rate will follow cosine schedule
- Token accuracy should improve over time
- GPU memory usage should remain stable

## 🔍 Troubleshooting

### Issue: "No metrics data available"
**Solution**: The app now automatically recovers experiments from backup

### Issue: Plots not showing
**Solution**: 
1. Check experiment ID is correct
2. Try different metrics (loss, learning_rate, etc.)
3. Refresh the page

### Issue: Data not persisting
**Solution**: 
1. App now uses `/tmp/` for better persistence
2. Backup recovery ensures data availability
3. Multiple storage locations provide redundancy

## 🎯 Next Steps

1. **Deploy Updated App**: Push the updated `app.py` to your HF Space
2. **Test Plots**: Try plotting your experiments
3. **Monitor Training**: Continue monitoring your training runs
4. **Add New Experiments**: Create new experiments as needed

## 📞 Support

If you encounter issues:
1. Check the logs in your HF Space
2. Verify experiment IDs are correct
3. Try the backup recovery feature
4. Contact for additional support

---

**Your experiments are now properly configured and should display correctly in the Trackio interface!** 🎉 