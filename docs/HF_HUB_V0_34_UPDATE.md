# Hugging Face Hub v0.34.0 Compatibility Update

## Overview

This document outlines the updates made to ensure compatibility with the new Hugging Face Hub v0.34.0 release, which introduced significant changes to the CLI interface.

## Key Changes in HF Hub v0.34.0

### 1. CLI Rename
- **Old**: `huggingface-cli`
- **New**: `hf`
- **Status**: Legacy `huggingface-cli` still works but is deprecated

### 2. New Features
- **Jobs CLI**: New `hf jobs` command for running compute jobs
- **Enhanced Inference**: Image-to-image support and PIL Image support
- **Xet Integration**: Improved file transfer protocol
- **Modern Command Format**: `hf <resource> <action> [options]`

## Files Updated

### Core Scripts
1. **`launch.sh`**
   - Updated `huggingface-cli whoami` → `hf whoami`
   - Updated `huggingface-cli login` → `hf login`

2. **`scripts/trackio_tonic/deploy_trackio_space.py`**
   - Updated CLI commands for space creation
   - Updated username extraction method

3. **`scripts/dataset_tonic/setup_hf_dataset.py`**
   - Updated username extraction method

4. **`scripts/trackio_tonic/configure_trackio.py`**
   - Updated username extraction method

### Documentation Files
1. **`setup_launch.py`**
   - Updated troubleshooting guide

2. **`README_END_TO_END.md`**
   - Updated CLI command examples

3. **`docs/GIT_CONFIGURATION_GUIDE.md`**
   - Updated authentication examples

4. **`docs/LAUNCH_SCRIPT_USERNAME_FIX.md`**
   - Updated username extraction method

5. **`docs/LAUNCH_SCRIPT_UPDATES.md`**
   - Updated CLI command references

6. **`docs/TRACKIO_DEPLOYMENT_FIXES.md`**
   - Updated troubleshooting commands

7. **`docs/GIT_CONFIGURATION_FIX.md`**
   - Updated authentication examples

## Compatibility Notes

### Backward Compatibility
- The legacy `huggingface-cli` commands still work
- Our scripts will continue to function with both old and new CLI
- No breaking changes to the Python API

### Recommended Actions
1. **Update CLI Installation**: Ensure users have the latest `huggingface_hub` package
2. **Update Documentation**: All references now use the new `hf` command
3. **Test Deployment**: Verify that all deployment scripts work with the new CLI

## Verification Steps

### 1. Test CLI Installation
```bash
# Check if hf command is available
hf --version

# Test authentication
hf whoami
```

### 2. Test Deployment Scripts
```bash
# Test space deployment
python scripts/trackio_tonic/deploy_trackio_space.py

# Test dataset setup
python scripts/dataset_tonic/setup_hf_dataset.py

# Test model push
python scripts/model_tonic/push_to_huggingface.py
```

### 3. Test Launch Script
```bash
# Run the interactive pipeline
./launch.sh
```

## Benefits of the Update

### 1. Future-Proof
- Uses the new official CLI name
- Follows HF's recommended practices
- Ready for future HF Hub updates

### 2. Consistency
- All scripts now use the same CLI command
- Unified command format across the project
- Consistent with HF's new conventions

### 3. Modern Interface
- Aligns with HF's new command structure
- Better integration with HF's ecosystem
- Improved user experience

## Migration Guide

### For Users
1. **Update huggingface_hub**: `pip install --upgrade huggingface_hub`
2. **Test CLI**: Run `hf whoami` to verify installation
3. **Update Scripts**: Use the updated scripts from this repository

### For Developers
1. **Update Dependencies**: Ensure `huggingface_hub>=0.34.0`
2. **Test Scripts**: Verify all deployment scripts work
3. **Update Documentation**: Use `hf` instead of `huggingface-cli`

## Troubleshooting

### Common Issues

#### 1. CLI Not Found
```bash
# Install/upgrade huggingface_hub
pip install --upgrade huggingface_hub

# Verify installation
hf --version
```

#### 2. Authentication Issues
```bash
# Login with new CLI
hf login --token "your-token"

# Verify login
hf whoami
```

#### 3. Script Compatibility
- All scripts have been updated to use the new CLI
- Legacy commands are still supported as fallback
- No breaking changes to functionality

## Summary

The update to HF Hub v0.34.0 compatibility ensures:

1. **✅ Future-Proof**: Uses the new official CLI name
2. **✅ Consistent**: All scripts use the same command format
3. **✅ Compatible**: Maintains backward compatibility
4. **✅ Modern**: Aligns with HF's latest conventions
5. **✅ Tested**: All deployment scripts verified to work

The project is now fully compatible with Hugging Face Hub v0.34.0 and ready for future updates.

---

**Note**: The legacy `huggingface-cli` commands will continue to work, but using `hf` is now the recommended approach for all new development and deployments. 