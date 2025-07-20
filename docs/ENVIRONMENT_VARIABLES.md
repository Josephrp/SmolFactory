# 🔧 Trackio Environment Variables Reference

## Quick Setup

Set these environment variables in your Hugging Face Space:

```bash
# Required: Your HF token for dataset access
HF_TOKEN=your_hf_token_here

# Optional: Dataset repository to use (defaults to tonic/trackio-experiments)
TRACKIO_DATASET_REPO=your-username/your-dataset-name
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ Yes | None | Your Hugging Face token for dataset access |
| `TRACKIO_DATASET_REPO` | ❌ No | `tonic/trackio-experiments` | Dataset repository to load experiments from |
| `SPACE_ID` | 🔄 Auto | None | HF Space ID (automatically detected) |

## Configuration Examples

### 1. Default Setup
```bash
HF_TOKEN=your_token_here
# Uses: tonic/trackio-experiments
```

### 2. Personal Dataset
```bash
HF_TOKEN=your_token_here
TRACKIO_DATASET_REPO=your-username/trackio-experiments
```

### 3. Team Dataset
```bash
HF_TOKEN=your_token_here
TRACKIO_DATASET_REPO=your-org/team-experiments
```

### 4. Project-Specific Dataset
```bash
HF_TOKEN=your_token_here
TRACKIO_DATASET_REPO=your-username/smollm3-experiments
```

## How to Set in HF Spaces

1. Go to your Hugging Face Space settings
2. Navigate to "Settings" → "Environment variables"
3. Add the variables:
   - `HF_TOKEN`: Your HF token
   - `TRACKIO_DATASET_REPO`: Your dataset repository (optional)

## Testing Configuration

Run the configuration script to check your setup:

```bash
python configure_trackio.py
```

This will:
- ✅ Show current environment variables
- 🧪 Test dataset access
- 📊 Display experiment count
- 💾 Generate configuration file

## Getting Your HF Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "Trackio Access")
4. Select "Write" permissions
5. Copy the token and set it as `HF_TOKEN`

## Dataset Repository Format

The `TRACKIO_DATASET_REPO` should follow this format:
```
username/dataset-name
```

Examples:
- `tonic/trackio-experiments`
- `your-username/my-experiments`
- `your-org/team-experiments`

## Troubleshooting

### Issue: "HF_TOKEN not found"
**Solution**: Set your HF token in the Space environment variables

### Issue: "Failed to load dataset"
**Solutions**:
1. Check your token has read access to the dataset
2. Verify the dataset repository exists
3. Try the backup fallback (automatic)

### Issue: "Failed to save experiments"
**Solutions**:
1. Check your token has write permissions
2. Verify the dataset repository exists
3. Check network connectivity

## Security Notes

- 🔒 Dataset is private by default
- 🔑 Only accessible with your HF_TOKEN
- 🛡️ No sensitive data exposed publicly
- 🔐 Secure storage on HF infrastructure 