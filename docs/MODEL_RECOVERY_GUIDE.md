# Model Recovery and Deployment Guide

This guide will help you recover your trained model from the cloud instance and deploy it to Hugging Face Hub with quantization.

## Prerequisites

1. **Hugging Face Token**: You need a Hugging Face token with write permissions
2. **Cloud Instance Access**: SSH access to your cloud instance
3. **Model Files**: Your trained model should be in `/output-checkpoint/` on the cloud instance

## Step 1: Connect to Your Cloud Instance

```bash
ssh root@your-cloud-instance-ip
cd ~/smollm3_finetune
```

## Step 2: Set Your Hugging Face Token

```bash
export HF_TOKEN=your_huggingface_token_here
```

Replace `your_huggingface_token_here` with your actual Hugging Face token.

## Step 3: Verify Model Files

Check that your model files exist:

```bash
ls -la /output-checkpoint/
```

You should see files like:
- `config.json`
- `model.safetensors.index.json`
- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

## Step 4: Update Configuration

Edit the deployment script to use your Hugging Face username:

```bash
nano cloud_deploy.py
```

Change this line:
```python
REPO_NAME = "your-username/smollm3-finetuned"  # Change to your HF username and desired repo name
```

To your actual username, for example:
```python
REPO_NAME = "tonic/smollm3-finetuned"
```

## Step 5: Run the Deployment

Execute the deployment script:

```bash
python3 cloud_deploy.py
```

This will:
1. ✅ Validate your model files
2. ✅ Install required dependencies (torchao, huggingface_hub)
3. ✅ Push the main model to Hugging Face Hub
4. ✅ Create quantized versions (int8 and int4)
5. ✅ Push quantized models to subdirectories

## Step 6: Verify Deployment

After successful deployment, you can verify:

1. **Main Model**: https://huggingface.co/your-username/smollm3-finetuned
2. **int8 Quantized**: https://huggingface.co/your-username/smollm3-finetuned/int8
3. **int4 Quantized**: https://huggingface.co/your-username/smollm3-finetuned/int4

## Alternative: Manual Deployment

If you prefer to run the steps manually:

### 1. Push Main Model Only

```bash
python3 scripts/model_tonic/push_to_huggingface.py \
    /output-checkpoint/ \
    your-username/smollm3-finetuned \
    --hf-token $HF_TOKEN \
    --author-name "Your Name" \
    --model-description "A fine-tuned SmolLM3 model for improved text generation"
```

### 2. Quantize and Push (Optional)

```bash
# int8 quantization (GPU optimized)
python3 scripts/model_tonic/quantize_model.py \
    /output-checkpoint/ \
    your-username/smollm3-finetuned \
    --quant-type int8_weight_only \
    --hf-token $HF_TOKEN

# int4 quantization (CPU optimized)
python3 scripts/model_tonic/quantize_model.py \
    /output-checkpoint/ \
    your-username/smollm3-finetuned \
    --quant-type int4_weight_only \
    --hf-token $HF_TOKEN
```

## Troubleshooting

### Common Issues

1. **HF_TOKEN not set**
   ```bash
   export HF_TOKEN=your_token_here
   ```

2. **Model files not found**
   ```bash
   ls -la /output-checkpoint/
   ```
   Make sure the training completed successfully.

3. **Dependencies missing**
   ```bash
   pip install torchao huggingface_hub
   ```

4. **Permission denied**
   ```bash
   chmod +x cloud_deploy.py
   chmod +x recover_model.py
   ```

### Error Messages

- **"Missing required model files"**: Check that your model training completed successfully
- **"Repository creation failed"**: Verify your HF token has write permissions
- **"Quantization failed"**: Check GPU memory availability or try CPU quantization

## Model Usage

Once deployed, you can use your model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Main model
model = AutoModelForCausalLM.from_pretrained("your-username/smollm3-finetuned")
tokenizer = AutoTokenizer.from_pretrained("your-username/smollm3-finetuned")

# int8 quantized (GPU optimized)
model = AutoModelForCausalLM.from_pretrained("your-username/smollm3-finetuned/int8")
tokenizer = AutoTokenizer.from_pretrained("your-username/smollm3-finetuned/int8")

# int4 quantized (CPU optimized)
model = AutoModelForCausalLM.from_pretrained("your-username/smollm3-finetuned/int4")
tokenizer = AutoTokenizer.from_pretrained("your-username/smollm3-finetuned/int4")

# Generate text
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## File Structure

After deployment, your repository will have:

```
your-username/smollm3-finetuned/
├── README.md (model card)
├── config.json
├── model.safetensors.index.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── int8/ (quantized model for GPU)
│   ├── README.md
│   ├── config.json
│   └── pytorch_model.bin
└── int4/ (quantized model for CPU)
    ├── README.md
    ├── config.json
    └── pytorch_model.bin
```

## Success Indicators

✅ **Successful deployment shows:**
- "Model recovery and deployment completed successfully!"
- "View your model at: https://huggingface.co/your-username/smollm3-finetuned"
- No error messages in the output

❌ **Failed deployment shows:**
- Error messages about missing files or permissions
- "Model recovery and deployment failed!"

## Next Steps

After successful deployment:

1. **Test your model** on Hugging Face Hub
2. **Share your model** with the community
3. **Monitor usage** through Hugging Face analytics
4. **Consider fine-tuning** further based on feedback

## Support

If you encounter issues:

1. Check the error messages carefully
2. Verify your HF token permissions
3. Ensure all model files are present
4. Try running individual steps manually
5. Check the logs for detailed error information

---

**Happy deploying! 🚀**