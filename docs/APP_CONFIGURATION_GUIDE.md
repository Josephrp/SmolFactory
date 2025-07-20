# ⚙️ App Configuration Guide

## Overview

The Trackio app now includes a **Configuration tab** that allows you to set your Hugging Face token and dataset repository directly through the interface, providing an alternative to environment variables.

## 🚀 New Features

### **Configuration Tab**
- ✅ **HF Token Input**: Secure password field for your Hugging Face token
- ✅ **Dataset Repository Input**: Text field for your dataset repository
- ✅ **Update Configuration**: Apply new settings and reload experiments
- ✅ **Test Connection**: Verify access to the dataset repository
- ✅ **Create Dataset**: Create a new dataset repository if it doesn't exist

### **Flexible Configuration**
- ✅ **Environment Variables**: Still supported as fallback
- ✅ **Interface Input**: New direct input method
- ✅ **Dynamic Updates**: Change configuration without restarting
- ✅ **Validation**: Input validation and error handling

## 📋 Configuration Tab Usage

### **1. Access the Configuration Tab**
- Open the Trackio app
- Click on the "⚙️ Configuration" tab
- You'll see input fields for HF Token and Dataset Repository

### **2. Set Your HF Token**
```
Hugging Face Token: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
- **Type**: Password field (hidden for security)
- **Required**: Yes (for dataset access)
- **Format**: Your HF token starting with `hf_`
- **Help**: Click the help text for instructions on getting your token

### **3. Set Your Dataset Repository**
```
Dataset Repository: your-username/your-dataset-name
```
- **Type**: Text field
- **Required**: No (defaults to `tonic/trackio-experiments`)
- **Format**: `username/dataset-name`
- **Examples**: 
  - `tonic/trackio-experiments`
  - `your-username/my-experiments`
  - `your-org/team-experiments`

### **4. Use the Action Buttons**

#### **Update Configuration**
- Applies new settings immediately
- Reloads experiments with new configuration
- Shows current status and experiment count

#### **Test Connection**
- Verifies access to the dataset repository
- Tests HF token permissions
- Shows dataset information and experiment count

#### **Create Dataset**
- Creates a new dataset repository if it doesn't exist
- Sets up the correct schema for experiments
- Makes the dataset private by default

## 🔧 Configuration Methods

### **Method 1: Interface Input (New)**
1. Go to "⚙️ Configuration" tab
2. Enter your HF token and dataset repository
3. Click "Update Configuration"
4. Verify with "Test Connection"

### **Method 2: Environment Variables (Existing)**
```bash
# Set environment variables
export HF_TOKEN=your_hf_token_here
export TRACKIO_DATASET_REPO=your-username/your-dataset-name

# Or for HF Spaces, add to Space settings
HF_TOKEN=your_hf_token_here
TRACKIO_DATASET_REPO=your-username/your-dataset-name
```

### **Method 3: Hybrid Approach**
- Set environment variables as defaults
- Override specific values through the interface
- Interface values take precedence over environment variables

## 📊 Configuration Priority

The app uses this priority order for configuration:

1. **Interface Input** (highest priority)
2. **Environment Variables** (fallback)
3. **Default Values** (lowest priority)

## 🛠️ Getting Your HF Token

### **Step-by-Step Instructions**
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "Trackio Access")
4. Select "Write" permissions
5. Click "Generate token"
6. Copy the token (starts with `hf_`)
7. Paste it in the app's HF Token field

### **Token Permissions**
- **Read**: Required for loading experiments
- **Write**: Required for saving experiments
- **Scope**: Should have access to your dataset repositories

## 📁 Dataset Repository Format

### **Correct Format**
```
username/dataset-name
```

### **Examples**
- `tonic/trackio-experiments` (default)
- `your-username/my-experiments`
- `your-org/team-experiments`
- `your-username/smollm3-experiments`

### **Validation**
- Must contain exactly one `/`
- Username must be valid HF username
- Dataset name must be valid (alphanumeric + hyphens)

## 🔍 Testing Your Configuration

### **1. Test Connection**
- Enter your HF token and dataset repository
- Click "Test Connection"
- Should show: "✅ Connection successful!"

### **2. Create Dataset (if needed)**
- If dataset doesn't exist, click "Create Dataset"
- Should show: "✅ Dataset created successfully!"

### **3. Update Configuration**
- Click "Update Configuration"
- Should show: "✅ Configuration updated successfully!"

## 🚨 Troubleshooting

### **Issue: "Please provide a Hugging Face token"**
**Solution**: 
- Enter your HF token in the interface
- Or set the `HF_TOKEN` environment variable

### **Issue: "Connection failed: 401 Unauthorized"**
**Solutions**:
1. Check your HF token is correct
2. Verify the token has read access to the dataset
3. Ensure the dataset repository exists

### **Issue: "Failed to create dataset"**
**Solutions**:
1. Check your HF token has write permissions
2. Verify the username in the repository name
3. Ensure the dataset name is valid

### **Issue: "Dataset repository must be in format: username/dataset-name"**
**Solution**: 
- Use the correct format: `username/dataset-name`
- Example: `your-username/my-experiments`

## 📈 Benefits

### **For Users**
- ✅ **Easy Setup**: No need to set environment variables
- ✅ **Visual Interface**: Clear input fields and validation
- ✅ **Immediate Feedback**: Test connection and see results
- ✅ **Flexible**: Can change configuration anytime

### **For Development**
- ✅ **Backward Compatible**: Environment variables still work
- ✅ **Fallback Support**: Graceful degradation
- ✅ **Error Handling**: Clear error messages
- ✅ **Validation**: Input validation and testing

### **For Deployment**
- ✅ **HF Spaces Ready**: Works on Hugging Face Spaces
- ✅ **No Restart Required**: Dynamic configuration updates
- ✅ **Secure**: Password field for token input
- ✅ **User-Friendly**: Clear instructions and help text

## 🎯 Usage Examples

### **Basic Setup**
1. Open the app
2. Go to "⚙️ Configuration" tab
3. Enter your HF token
4. Enter your dataset repository
5. Click "Update Configuration"
6. Click "Test Connection" to verify

### **Advanced Setup**
1. Set environment variables as defaults
2. Use interface to override specific values
3. Test connection to verify access
4. Create dataset if it doesn't exist
5. Start using the app with persistent storage

### **Team Setup**
1. Create a shared dataset repository
2. Share the repository name with team
3. Each team member sets their own HF token
4. All experiments are stored in the shared dataset

## 📋 Configuration Status

The app shows current configuration status:
```
📊 Dataset: your-username/your-dataset
🔑 HF Token: Set
📈 Experiments: 5
```

## 🔄 Updating Configuration

You can update configuration at any time:
1. Go to "⚙️ Configuration" tab
2. Change HF token or dataset repository
3. Click "Update Configuration"
4. Experiments will reload with new settings

---

**🎉 Your Trackio app is now more flexible and user-friendly with direct configuration input!** 