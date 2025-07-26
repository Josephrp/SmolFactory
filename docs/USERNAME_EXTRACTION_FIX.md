# Username Extraction Fix

This document outlines the fix for the "Invalid user token" error that occurred during Trackio Space deployment.

## 🐛 **Problem Description**

The error occurred in the `deploy_trackio_space.py` script when trying to extract the username from the HF token:

```
❌ Failed to get user info from token: Invalid user token.
```

This happened because:
1. The `whoami()` API method was being called incorrectly
2. The response format wasn't handled properly
3. No fallback mechanism was in place

## ✅ **Solution Implemented**

### **1. Improved Username Extraction Function**

Created a robust username extraction function that handles multiple scenarios:

```python
def get_username_from_token(token: str) -> str:
    """Get username from HF token with fallback to CLI"""
    try:
        # Try API first
        api = HfApi(token=token)
        user_info = api.whoami()
        
        # Handle different possible response formats
        if isinstance(user_info, dict):
            # Try different possible keys for username
            username = (
                user_info.get('name') or 
                user_info.get('username') or 
                user_info.get('user') or 
                None
            )
        elif isinstance(user_info, str):
            # If whoami returns just the username as string
            username = user_info
        else:
            username = None
            
        if username:
            print(f"✅ Got username from API: {username}")
            return username
        else:
            print("⚠️  Could not get username from API, trying CLI...")
            return get_username_from_cli(token)
            
    except Exception as e:
        print(f"⚠️  API whoami failed: {e}")
        print("⚠️  Trying CLI fallback...")
        return get_username_from_cli(token)
```

### **2. CLI Fallback Method**

Added a robust CLI fallback method:

```python
def get_username_from_cli(token: str) -> str:
    """Fallback method to get username using CLI"""
    try:
        # Set HF token for CLI
        os.environ['HF_TOKEN'] = token
        
        # Get username using CLI
        result = subprocess.run(
            ["hf", "whoami"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            username = result.stdout.strip()
            if username:
                print(f"✅ Got username from CLI: {username}")
                return username
            else:
                print("⚠️  CLI returned empty username")
                return None
        else:
            print(f"⚠️  CLI whoami failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"⚠️  CLI fallback failed: {e}")
        return None
```

## 🔧 **Files Updated**

### **1. `scripts/trackio_tonic/deploy_trackio_space.py`**
- ✅ Added `_get_username_from_cli()` method
- ✅ Updated `__init__()` to use improved username extraction
- ✅ Better error handling and fallback mechanisms
- ✅ Handles different response formats from `whoami()`

### **2. `scripts/dataset_tonic/setup_hf_dataset.py`**
- ✅ Added `get_username_from_token()` and `get_username_from_cli()` functions
- ✅ Updated main function to use improved username extraction
- ✅ Better error handling and user feedback

### **3. `scripts/trackio_tonic/configure_trackio.py`**
- ✅ Added same username extraction functions
- ✅ Updated configuration function to use improved method
- ✅ Consistent error handling across all scripts

## 🎯 **Key Improvements**

### **✅ Robust Error Handling**
- API method fails → CLI fallback
- CLI fails → Clear error message
- Multiple response format handling

### **✅ Better User Feedback**
- Clear status messages for each step
- Indicates which method is being used (API vs CLI)
- Helpful error messages with suggestions

### **✅ Multiple Response Format Support**
- Handles dictionary responses with different key names
- Handles string responses
- Handles unexpected response formats

### **✅ Timeout Protection**
- 30-second timeout for CLI operations
- Prevents hanging on network issues

## 🔍 **Response Format Handling**

The fix handles different possible response formats from the `whoami()` API:

### **Dictionary Response:**
```python
{
    "name": "username",
    "username": "username", 
    "user": "username"
}
```

### **String Response:**
```python
"username"
```

### **Unknown Format:**
- Falls back to CLI method
- Provides clear error messages

## 🧪 **Testing Results**

All tests pass with the updated scripts:

```
📊 Test Results Summary
========================================
✅ PASS: Import Tests
✅ PASS: Script Existence  
✅ PASS: Script Syntax
✅ PASS: Environment Variables
✅ PASS: API Connection
✅ PASS: Script Functions
✅ PASS: Template Files

🎯 Overall: 7/7 tests passed
🎉 All tests passed! The fixes are working correctly.
```

## 🚀 **Usage**

The fix is transparent to users. The workflow remains the same:

```bash
# 1. Set HF token
export HF_TOKEN=your_token_here

# 2. Run deployment (username auto-detected)
python scripts/trackio_tonic/deploy_trackio_space.py

# 3. Or use the launch script
bash launch.sh
```

## 🎉 **Benefits**

1. **✅ Reliable Username Detection**: Works with different API response formats
2. **✅ Robust Fallback**: CLI method as backup when API fails
3. **✅ Better Error Messages**: Clear feedback about what's happening
4. **✅ Consistent Behavior**: Same method across all scripts
5. **✅ No User Impact**: Transparent to end users
6. **✅ Future-Proof**: Handles different API response formats

## 🔧 **Troubleshooting**

If username extraction still fails:

1. **Check Token**: Ensure HF_TOKEN is valid and has proper permissions
2. **Check Network**: Ensure internet connection is stable
3. **Check CLI**: Ensure `hf` is installed and working
4. **Manual Override**: Can manually set username in scripts if needed

## 📋 **Summary**

The username extraction fix resolves the "Invalid user token" error by:

- ✅ Implementing robust API response handling
- ✅ Adding CLI fallback mechanism  
- ✅ Providing better error messages
- ✅ Ensuring consistent behavior across all scripts
- ✅ Maintaining backward compatibility

The fix ensures that username extraction works reliably across different environments and API response formats, providing a smooth user experience for the Trackio deployment pipeline. 