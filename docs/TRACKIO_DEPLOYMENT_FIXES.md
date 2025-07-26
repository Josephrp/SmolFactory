# Trackio Deployment Fixes

This document outlines the fixes made to resolve the Trackio Space deployment and dataset creation issues.

## Issues Identified

### 1. Git Authentication Issues in Space Deployment
- **Problem**: The `deploy_trackio_space.py` script was using git commands for file upload, which failed with authentication errors
- **Solution**: Replaced git commands with direct HF Hub API calls using `upload_file()`

### 2. Dataset Repository Creation Issues
- **Problem**: The `setup_hf_dataset.py` script was trying to push to a dataset repository that didn't exist, causing 404 errors
- **Solution**: Added proper repository creation using `create_repo()` before pushing the dataset

### 3. Missing Environment Variable Setup
- **Problem**: The Space deployment didn't set up the required `HF_TOKEN` environment variable
- **Solution**: Added automatic secret setting using `add_space_secret()` API method

### 4. Manual Username Input Required
- **Problem**: Users had to manually enter their username
- **Solution**: Automatically extract username from token using `whoami()` API method

### 5. Dataset Access Testing Issues
- **Problem**: The configuration script failed when testing dataset access for non-existent datasets
- **Solution**: Added proper error handling and repository existence checks

## Fixed Scripts

### 1. `scripts/trackio_tonic/deploy_trackio_space.py`

#### Key Changes:
- **Replaced git upload with HF Hub API**: Now uses `upload_file()` directly instead of git commands
- **Automatic secret setting**: Uses `add_space_secret()` API to set HF_TOKEN automatically
- **Username extraction from token**: Uses `whoami()` to get username automatically
- **Removed manual username input**: No longer asks for username
- **Improved error handling**: Better error messages and fallback options

#### Usage:
```bash
python scripts/trackio_tonic/deploy_trackio_space.py
```

#### What it does:
1. Extracts username from HF token automatically
2. Creates a new HF Space using the API
3. Prepares Space files from templates
4. Uploads files using HF Hub API (no git required)
5. **Automatically sets secrets via API** (HF_TOKEN and TRACKIO_DATASET_REPO)
6. Tests the Space accessibility

### 2. `scripts/dataset_tonic/setup_hf_dataset.py`

#### Key Changes:
- **Added repository creation**: Creates the dataset repository before pushing data
- **Username extraction from token**: Uses `whoami()` to get username automatically
- **Automatic dataset naming**: Uses username in dataset repository name
- **Improved error handling**: Better error messages for common issues
- **Public datasets by default**: Makes datasets public for easier access

#### Usage:
```bash
python scripts/dataset_tonic/setup_hf_dataset.py
```

#### What it does:
1. Extracts username from HF token automatically
2. Creates the dataset repository if it doesn't exist
3. Creates a dataset with sample experiment data
4. Uploads README template
5. Makes the dataset public for easier access

### 3. `scripts/trackio_tonic/configure_trackio.py`

#### Key Changes:
- **Added repository existence check**: Checks if dataset repository exists before trying to load
- **Username extraction from token**: Uses `whoami()` to get username automatically
- **Automatic dataset naming**: Uses username in default dataset repository
- **Better error handling**: Distinguishes between missing repository and permission issues
- **Improved user guidance**: Clear instructions for next steps

#### Usage:
```bash
python scripts/trackio_tonic/configure_trackio.py
```

#### What it does:
1. Extracts username from HF token automatically
2. Validates current configuration
3. Tests dataset access with proper error handling
4. Generates configuration file with username
5. Provides usage examples with actual username

## Model Push Script (`scripts/model_tonic/push_to_huggingface.py`)

The model push script was already using the HF Hub API correctly, so no changes were needed. It properly:
- Creates repositories using `create_repo()`
- Uploads files using `upload_file()`
- Handles authentication correctly

## Environment Variables Required

### For HF Spaces:
```bash
HF_TOKEN=your_hf_token_here
TRACKIO_DATASET_REPO=your-username/your-dataset-name
```

### For Local Development:
```bash
export HF_TOKEN=your_hf_token_here
export TRACKIO_DATASET_REPO=your-username/your-dataset-name
```

## Deployment Workflow

### 1. Create Dataset
```bash
# Set environment variables
export HF_TOKEN=your_token_here
# TRACKIO_DATASET_REPO will be auto-generated as username/trackio-experiments

# Create the dataset
python scripts/dataset_tonic/setup_hf_dataset.py
```

### 2. Deploy Trackio Space
```bash
# Deploy the Space (no username needed - extracted from token)
python scripts/trackio_tonic/deploy_trackio_space.py
```

### 3. Secrets are Automatically Set
The script now automatically sets the required secrets via the HF Hub API:
- `HF_TOKEN` - Your Hugging Face token
- `TRACKIO_DATASET_REPO` - Your dataset repository (if specified)

### 4. Test Configuration
```bash
# Test the configuration
python scripts/trackio_tonic/configure_trackio.py
```

## New Features

### ✅ **Automatic Secret Setting**
- Uses `add_space_secret()` API method
- Sets `HF_TOKEN` automatically
- Sets `TRACKIO_DATASET_REPO` if specified
- Falls back to manual instructions if API fails

### ✅ **Username Extraction from Token**
- Uses `whoami()` API method
- No manual username input required
- Automatically uses username in dataset names
- Provides better user experience

### ✅ **Improved User Experience**
- Fewer manual inputs required
- Automatic configuration based on token
- Clear feedback about what's happening
- Better error messages

## Troubleshooting

### Common Issues:

1. **"Repository not found" errors**:
   - Run `setup_hf_dataset.py` to create the dataset first
   - Check that your HF token has write permissions

2. **"Authentication failed" errors**:
   - Verify your HF token is valid
   - Check token permissions on https://huggingface.co/settings/tokens

3. **"Space not accessible" errors**:
   - Wait 2-5 minutes for the Space to build
   - Check Space logs at the Space URL
   - Verify all files were uploaded correctly

4. **"Dataset access failed" errors**:
   - Ensure the dataset repository exists
   - Check that your token has read permissions
   - Verify the dataset repository name is correct

5. **"Secret setting failed" errors**:
   - The script will fall back to manual instructions
   - Follow the provided instructions to set secrets manually
   - Check that your token has write permissions to the Space

### Debugging Steps:

1. **Check token permissions**:
   ```bash
   hf whoami
   ```

2. **Test dataset access**:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("your-username/your-dataset", token="your-token")
   ```

3. **Test Space deployment**:
   ```bash
   python scripts/trackio_tonic/deploy_trackio_space.py
   ```

4. **Test secret setting**:
   ```python
   from huggingface_hub import HfApi
   api = HfApi(token="your-token")
   api.add_space_secret("your-username/your-space", "TEST_KEY", "test_value")
   ```

## Security Considerations

- **Public datasets**: Datasets are now public by default for easier access
- **Token security**: Never commit tokens to version control
- **Space secrets**: Automatically set via API, with manual fallback
- **Access control**: Verify token permissions before deployment

## Performance Improvements

- **Direct API calls**: Eliminated git dependency for faster uploads
- **Automatic configuration**: No manual username input required
- **Parallel processing**: Files are uploaded individually for better error handling
- **Caching**: HF Hub API handles caching automatically
- **Error recovery**: Better error handling and retry logic

## Future Enhancements

1. **Batch secret setting**: Set multiple secrets in one API call
2. **Progress tracking**: Add progress bars for large uploads
3. **Validation**: Add more comprehensive validation checks
4. **Rollback**: Add ability to rollback failed deployments
5. **Hardware configuration**: Automatically configure Space hardware

## Testing

To test the fixes:

```bash
# Test dataset creation
python scripts/dataset_tonic/setup_hf_dataset.py

# Test Space deployment
python scripts/trackio_tonic/deploy_trackio_space.py

# Test configuration
python scripts/trackio_tonic/configure_trackio.py

# Test model push (if you have a trained model)
python scripts/model_tonic/push_to_huggingface.py --model-path /path/to/model --repo-name your-username/your-model
```

## Summary

These fixes resolve the main issues with:
- ✅ Git authentication problems
- ✅ Dataset repository creation failures
- ✅ Missing environment variable setup
- ✅ Manual username input requirement
- ✅ Poor error handling and user feedback
- ✅ Security concerns with public datasets

The scripts now use the HF Hub API directly, provide better error messages, handle edge cases properly, and offer a much improved user experience with automatic configuration. 