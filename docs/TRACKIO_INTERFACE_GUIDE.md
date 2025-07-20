# Enhanced Trackio Interface Guide

## Overview

Your Trackio application has been significantly enhanced to provide comprehensive monitoring and visualization for SmolLM3 training experiments. Here's how to make the most of it.

## 🚀 Key Enhancements

### 1. **Real-time Visualization**
- **Interactive Plots**: Loss curves, accuracy, learning rate, GPU metrics
- **Experiment Comparison**: Compare multiple training runs side-by-side
- **Live Updates**: Watch training progress in real-time

### 2. **Comprehensive Data Display**
- **Formatted Output**: Clean, emoji-rich experiment details
- **Statistics Overview**: Metrics count, parameters count, artifacts count
- **Status Tracking**: Visual status indicators (🟢 running, ✅ completed, ❌ failed)

### 3. **Demo Data Generation**
- **Realistic Simulation**: Generate realistic training metrics for testing
- **Multiple Metrics**: Loss, accuracy, learning rate, GPU memory, training time
- **Configurable Parameters**: Customize demo data to match your setup

## 📊 How to Use with Your SmolLM3 Training

### Step 1: Start Your Training
```bash
python run_a100_large_experiment.py \
    --config config/train_smollm3_openhermes_fr_a100_balanced.py \
    --trackio_url "https://tonic-test-trackio-test.hf.space" \
    --experiment-name "petit-elle-l-aime-3-balanced" \
    --output-dir ./outputs/balanced
```

### Step 2: Monitor in Real-time
1. **Visit your Trackio Space**: `https://tonic-test-trackio-test.hf.space`
2. **Go to "View Experiments" tab**
3. **Enter your experiment ID** (e.g., `exp_20231201_143022`)
4. **Click "View Experiment"** to see detailed information

### Step 3: Visualize Training Progress
1. **Go to "📊 Visualizations" tab**
2. **Enter your experiment ID**
3. **Select a metric** (loss, accuracy, learning_rate, gpu_memory, training_time)
4. **Click "Create Plot"** to see interactive charts

### Step 4: Compare Experiments
1. **In the "📊 Visualizations" tab**
2. **Enter multiple experiment IDs** (comma-separated)
3. **Click "Compare Experiments"** to see side-by-side comparison

## 🎯 Interface Features

### Create Experiment Tab
- **Experiment Name**: Descriptive name for your training run
- **Description**: Detailed description of what you're training
- **Automatic ID Generation**: Unique experiment identifier

### Log Metrics Tab
- **Experiment ID**: The experiment to log metrics for
- **Metrics JSON**: Training metrics in JSON format
- **Step**: Current training step (optional)

Example metrics JSON:
```json
{
  "loss": 0.5234,
  "accuracy": 0.8567,
  "learning_rate": 3.5e-6,
  "gpu_memory_gb": 22.5,
  "gpu_utilization_percent": 87.3,
  "training_time_per_step": 0.456
}
```

### Log Parameters Tab
- **Experiment ID**: The experiment to log parameters for
- **Parameters JSON**: Training configuration in JSON format

Example parameters JSON:
```json
{
  "model_name": "HuggingFaceTB/SmolLM3-3B",
  "batch_size": 8,
  "learning_rate": 3.5e-6,
  "max_iters": 18000,
  "mixed_precision": "bf16",
  "no_think_system_message": true
}
```

### View Experiments Tab
- **Experiment ID**: Enter to view specific experiment
- **List All Experiments**: Shows overview of all experiments
- **Detailed Information**: Formatted display with statistics

### 📊 Visualizations Tab
- **Training Metrics**: Interactive plots for individual metrics
- **Experiment Comparison**: Side-by-side comparison of multiple runs
- **Real-time Updates**: Plots update as new data is logged

### 🎯 Demo Data Tab
- **Generate Demo Data**: Create realistic training data for testing
- **Configurable**: Adjust parameters to match your setup
- **Multiple Metrics**: Simulates loss, accuracy, GPU metrics, etc.

### Update Status Tab
- **Experiment ID**: The experiment to update
- **Status**: running, completed, failed, paused
- **Visual Indicators**: Status shown with emojis

## 📈 What Gets Displayed

### Training Metrics
- **Loss**: Training loss over time
- **Accuracy**: Model accuracy progression
- **Learning Rate**: Learning rate scheduling
- **GPU Memory**: Memory usage in GB
- **GPU Utilization**: GPU usage percentage
- **Training Time**: Time per training step

### Experiment Details
- **Basic Info**: ID, name, description, status, creation time
- **Statistics**: Metrics count, parameters count, artifacts count
- **Parameters**: All training configuration
- **Latest Metrics**: Most recent training metrics

### Visualizations
- **Line Charts**: Smooth curves showing metric progression
- **Interactive Hover**: Detailed information on hover
- **Multiple Metrics**: Switch between different metrics
- **Comparison Charts**: Side-by-side experiment comparison

## 🔧 Integration with Your Training

### Automatic Integration
Your training script automatically:
1. **Creates experiments** with your specified name
2. **Logs parameters** from your configuration
3. **Logs metrics** every 25 steps (configurable)
4. **Logs system metrics** (GPU memory, utilization)
5. **Logs checkpoints** every 2000 steps
6. **Updates status** when training completes

### Manual Integration
You can also manually:
1. **Create experiments** through the interface
2. **Log custom metrics** for specific analysis
3. **Compare different runs** with different parameters
4. **Generate demo data** for testing the interface

## 🎨 Customization

### Adding Custom Metrics
```python
# In your training script
custom_metrics = {
    "loss": current_loss,
    "accuracy": current_accuracy,
    "custom_metric": your_custom_value,
    "gpu_memory": gpu_memory_usage
}

monitor.log_metrics(custom_metrics, step=current_step)
```

### Custom Visualizations
The interface supports any metric you log. Just add it to your metrics JSON and it will appear in the visualization dropdown.

## 🚨 Troubleshooting

### No Data Displayed
1. **Check experiment ID**: Make sure you're using the correct ID
2. **Verify metrics were logged**: Check if training is actually logging metrics
3. **Use demo data**: Generate demo data to test the interface

### Plots Not Updating
1. **Refresh the page**: Sometimes plots need a refresh
2. **Check data format**: Ensure metrics are in the correct JSON format
3. **Verify step numbers**: Make sure step numbers are increasing

### Interface Not Loading
1. **Check dependencies**: Ensure plotly and pandas are installed
2. **Check Gradio version**: Use Gradio 4.0.0 or higher
3. **Check browser console**: Look for JavaScript errors

## 📊 Example Workflow

1. **Start Training**:
   ```bash
   python run_a100_large_experiment.py --experiment-name "my_experiment"
   ```

2. **Monitor Progress**:
   - Visit your Trackio Space
   - Go to "View Experiments"
   - Enter your experiment ID
   - Watch real-time updates

3. **Visualize Results**:
   - Go to "📊 Visualizations"
   - Select "loss" metric
   - Create plot to see training progress

4. **Compare Runs**:
   - Run multiple experiments with different parameters
   - Use "Compare Experiments" to see differences

5. **Generate Demo Data**:
   - Use "🎯 Demo Data" tab to test the interface
   - Generate realistic training data for demonstration

## 🎉 Success Indicators

Your interface is working correctly when you see:
- ✅ **Formatted experiment details** with emojis and structure
- ✅ **Interactive plots** that respond to your inputs
- ✅ **Real-time metric updates** during training
- ✅ **Clean experiment overview** with statistics
- ✅ **Smooth visualization** with hover information

The enhanced interface will now display much more meaningful information and provide a comprehensive monitoring experience for your SmolLM3 training experiments! 