# Hugging Face Token Validation Fix

## Problem Description

The original launch script was using the `hf` CLI command to validate Hugging Face tokens, which was causing authentication failures even with valid tokens. This was due to:

1. CLI installation issues
2. Inconsistent token format handling
3. Poor error reporting

## Solution Implementation

### New Python-Based Validation System

We've implemented a robust Python-based token validation system using the official `huggingface_hub` API:

#### Key Components

1. **`scripts/validate_hf_token.py`** - Main validation script
2. **Updated `launch.sh`** - Modified to use Python validation
3. **`tests/test_token_validation.py`** - Test suite for validation
4. **`scripts/check_dependencies.py`** - Dependency verification

### Features

- ✅ **Robust Error Handling**: Detailed error messages for different failure types
- ✅ **JSON Output**: Structured responses for easy parsing
- ✅ **Multiple Input Methods**: Command line arguments or environment variables
- ✅ **Username Extraction**: Automatically retrieves username from valid tokens
- ✅ **Dependency Checking**: Verifies required packages are installed

## Usage

### Direct Script Usage

```bash
# Using command line argument
python scripts/validate_hf_token.py hf_your_token_here

# Using environment variable
export HF_TOKEN=hf_your_token_here
python scripts/validate_hf_token.py
```

### Expected Output

**Success:**
```json
{"success": true, "username": "YourUsername", "error": null}
```

**Failure:**
```json
{"success": false, "username": null, "error": "Invalid token - unauthorized access"}
```

### Integration with Launch Script

The `launch.sh` script now automatically:

1. Prompts for your HF token
2. Validates it using the Python script
3. Extracts your username automatically
4. Provides detailed error messages if validation fails

## Error Types and Solutions

### Common Error Messages

| Error Message | Cause | Solution |
|---------------|-------|----------|
| "Invalid token - unauthorized access" | Token is invalid or expired | Generate new token at https://huggingface.co/settings/tokens |
| "Token lacks required permissions" | Token doesn't have write access | Ensure token has write permissions |
| "Network error" | Connection issues | Check internet connection |
| "Failed to run token validation script" | Missing dependencies | Run `pip install huggingface_hub` |

### Dependency Installation

```bash
# Install required dependencies
pip install huggingface_hub

# Check all dependencies
python scripts/check_dependencies.py

# Install all requirements
pip install -r requirements/requirements.txt
```

## Testing

### Run the Test Suite

```bash
python tests/test_token_validation.py
```

### Manual Testing

```bash
# Test with your token
python scripts/validate_hf_token.py hf_your_token_here

# Test dependency check
python scripts/check_dependencies.py
```

## Troubleshooting

### If Token Validation Still Fails

1. **Check Token Format**: Ensure token starts with `hf_`
2. **Verify Token Permissions**: Token needs read/write access
3. **Check Network**: Ensure internet connection is stable
4. **Update Dependencies**: Run `pip install --upgrade huggingface_hub`

### If Launch Script Fails

1. **Check Python Path**: Ensure `python3` is available
2. **Verify Script Permissions**: Script should be executable
3. **Check JSON Parsing**: Ensure Python can parse JSON output
4. **Review Error Messages**: Check the specific error in launch.sh output

## Technical Details

### Token Validation Process

1. **Environment Setup**: Sets `HUGGING_FACE_HUB_TOKEN` environment variable
2. **API Client Creation**: Initializes `HfApi()` client
3. **User Info Retrieval**: Calls `api.whoami()` to validate token
4. **Username Extraction**: Extracts username from user info
5. **Error Handling**: Catches and categorizes different error types

### JSON Parsing in Shell

The launch script uses Python's JSON parser to safely extract values:

```bash
local success=$(echo "$result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('success', False))
except:
    print('False')
")
```

## Migration from Old System

### Before (CLI-based)
```bash
if hf whoami >/dev/null 2>&1; then
    HF_USERNAME=$(hf whoami | head -n1 | tr -d '\n')
```

### After (Python-based)
```bash
if result=$(python3 scripts/validate_hf_token.py "$token" 2>/dev/null); then
    # Parse JSON result with error handling
    local success=$(echo "$result" | python3 -c "...")
    local username=$(echo "$result" | python3 -c "...")
```

## Benefits

1. **Reliability**: Uses official Python API instead of CLI
2. **Error Reporting**: Detailed error messages for debugging
3. **Cross-Platform**: Works on Windows, Linux, and macOS
4. **Maintainability**: Easy to update and extend
5. **Testing**: Comprehensive test suite included

## Future Enhancements

- [ ] Add token expiration checking
- [ ] Implement token refresh functionality
- [ ] Add support for organization tokens
- [ ] Create GUI for token management
- [ ] Add token security validation

---

**Note**: This fix ensures that valid Hugging Face tokens are properly recognized and that users get clear feedback when there are authentication issues. 