# SmolLM3 End-to-End Fine-tuning Pipeline

This repository provides a complete end-to-end pipeline for fine-tuning SmolLM3 models with integrated experiment tracking, monitoring, and model deployment.

## 🚀 Quick Start

### 1. Setup Configuration

```bash
# Run the setup script to configure with your information
python setup_launch.py
```

This will prompt you for:
- Your Hugging Face username
- Your Hugging Face token
- Optional model and dataset customizations

### 2. Check Requirements

```bash
# Verify all dependencies are installed
python check_requirements.py
```

### 3. Run the Pipeline

```bash
# Make the script executable and run
chmod +x launch.sh
./launch.sh
```

## 📋 What the Pipeline Does

The end-to-end pipeline performs the following steps:

### 1. **Environment Setup**
- Installs system dependencies
- Creates Python virtual environment
- Installs PyTorch with CUDA support
- Installs all required Python packages

### 2. **Trackio Space Deployment**
- Creates a new Hugging Face Space for experiment tracking
- Configures the Trackio monitoring interface
- Sets up environment variables

### 3. **HF Dataset Setup**
- Creates a Hugging Face Dataset repository for experiment storage
- Configures dataset access and permissions
- Sets up initial experiment data structure

### 4. **Dataset Preparation**
- Downloads the specified dataset from Hugging Face Hub
- Converts to training format (prompt/completion pairs)
- Handles multiple dataset formats automatically
- Creates train/validation splits

### 5. **Training Configuration**
- Creates optimized training configuration
- Sets up monitoring integration
- Configures model parameters and hyperparameters

### 6. **Model Training**
- Runs the SmolLM3 fine-tuning process
- Logs metrics to Trackio Space in real-time
- Saves experiment data to HF Dataset
- Creates checkpoints during training

### 7. **Model Deployment**
- Pushes trained model to Hugging Face Hub
- Creates comprehensive model card
- Uploads training results and logs
- Tests the uploaded model

### 8. **Summary Report**
- Generates detailed training summary
- Provides links to all resources
- Documents configuration and results

## 🎯 Features

### **Integrated Monitoring**
- Real-time experiment tracking via Trackio Space
- Persistent storage in Hugging Face Datasets
- Comprehensive metrics logging
- System resource monitoring

### **Flexible Dataset Support**
- Automatic format detection and conversion
- Support for multiple dataset types
- Built-in data preprocessing
- Train/validation split handling

### **Optimized Training**
- Flash Attention support for efficiency
- Gradient checkpointing for memory optimization
- Mixed precision training
- Automatic hyperparameter optimization

### **Complete Deployment**
- Automated model upload to Hugging Face Hub
- Comprehensive model cards
- Training results documentation
- Model testing and validation

## 📊 Monitoring & Tracking

### **Trackio Space Interface**
- Real-time training metrics visualization
- Experiment management and comparison
- System resource monitoring
- Training progress tracking

### **HF Dataset Storage**
- Persistent experiment data storage
- Version-controlled experiment history
- Collaborative experiment sharing
- Automated data backup

## 🔧 Configuration

### **Required Configuration**
Update these variables in `launch.sh`:

```bash
# Your Hugging Face credentials
HF_TOKEN="your_hf_token_here"
HF_USERNAME="your-username"

# Model and dataset
MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
DATASET_NAME="HuggingFaceTB/smoltalk"

# Output repositories
REPO_NAME="your-username/smollm3-finetuned-$(date +%Y%m%d)"
TRACKIO_DATASET_REPO="your-username/trackio-experiments"
```

### **Training Parameters**
Customize training parameters:

```bash
# Training configuration
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=5e-6
MAX_EPOCHS=3
MAX_SEQ_LENGTH=4096
```

## 📁 Output Structure

After running the pipeline, you'll have:

```
├── training_dataset/           # Prepared dataset
│   ├── train.json
│   └── validation.json
├── /output-checkpoint/         # Model checkpoints
│   ├── config.json
│   ├── pytorch_model.bin
│   └── training_results/
├── training.log               # Training logs
├── training_summary.md        # Summary report
└── config/train_smollm3_end_to_end.py  # Training config
```

## 🌐 Online Resources

The pipeline creates these online resources:

- **Model Repository**: `https://huggingface.co/your-username/smollm3-finetuned-YYYYMMDD`
- **Trackio Space**: `https://huggingface.co/spaces/your-username/trackio-monitoring-YYYYMMDD`
- **Experiment Dataset**: `https://huggingface.co/datasets/your-username/trackio-experiments`

## 🛠️ Troubleshooting

### **Common Issues**

1. **HF Token Issues**
   ```bash
   # Verify your token is correct
   huggingface-cli whoami
   ```

2. **CUDA Issues**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size or gradient accumulation
   BATCH_SIZE=1
   GRADIENT_ACCUMULATION_STEPS=16
   ```

4. **Dataset Issues**
   ```bash
   # Test dataset access
   python -c "from datasets import load_dataset; print(load_dataset('your-dataset'))"
   ```

### **Debug Mode**

Run individual components for debugging:

```bash
# Test Trackio deployment
cd scripts/trackio_tonic
python deploy_trackio_space.py

# Test dataset setup
cd scripts/dataset_tonic
python setup_hf_dataset.py

# Test training
python src/train.py config/train_smollm3_end_to_end.py --help
```

## 📚 Advanced Usage

### **Custom Datasets**

For custom datasets, ensure they have one of these formats:

```json
// Format 1: Prompt/Completion
{
  "prompt": "What is machine learning?",
  "completion": "Machine learning is..."
}

// Format 2: Instruction/Output
{
  "instruction": "Explain machine learning",
  "output": "Machine learning is..."
}

// Format 3: Chat format
{
  "messages": [
    {"role": "user", "content": "What is ML?"},
    {"role": "assistant", "content": "ML is..."}
  ]
}
```

### **Custom Models**

To use different models, update the configuration:

```bash
MODEL_NAME="microsoft/DialoGPT-medium"
MAX_SEQ_LENGTH=1024
```

### **Custom Training**

Modify training parameters in the generated config:

```python
# In config/train_smollm3_end_to_end.py
config = SmolLM3Config(
    learning_rate=1e-5,  # Custom learning rate
    max_iters=5000,      # Custom training steps
    # ... other parameters
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the pipeline
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for the excellent transformers library
- The SmolLM3 team for the base model
- The Trackio team for experiment tracking
- The open-source community for contributions

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs in `training.log`
3. Check the Trackio Space for monitoring data
4. Open an issue on GitHub

---

**Happy Fine-tuning! 🚀** 