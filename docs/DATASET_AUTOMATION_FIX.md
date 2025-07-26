# Dataset Configuration Automation Fix

## Problem Description

The original launch script required users to manually specify their username in the dataset repository name, which was:
1. **Error-prone**: Users had to remember their username
2. **Inconsistent**: Different users might use different naming conventions
3. **Manual**: Required extra steps in the setup process

## Solution Implementation

### Automatic Dataset Repository Creation

We've implemented a Python-based solution that automatically:

1. **Extracts username from token**: Uses the HF API to get the username from the validated token
2. **Creates dataset repository**: Automatically creates `username/trackio-experiments` or custom name
3. **Sets environment variables**: Automatically configures `TRACKIO_DATASET_REPO`
4. **Provides customization**: Allows users to customize the dataset name if desired

### Key Components

#### 1. **`scripts/dataset_tonic/setup_hf_dataset.py`** - Main Dataset Setup Script
- Automatically detects username from HF token
- Creates dataset repository with proper permissions
- Supports custom dataset names
- Sets environment variables for other scripts

#### 2. **Updated `launch.sh`** - Enhanced User Experience
- Automatically creates dataset repository
- Provides options for default or custom dataset names
- Fallback to manual input if automatic creation fails
- Clear user feedback and progress indicators

#### 3. **Python API Integration** - Consistent Authentication
- Uses `HfApi(token=token)` for direct token authentication
- Avoids environment variable conflicts
- Consistent error handling across all scripts

## Usage Examples

### Automatic Dataset Creation (Default)

```bash
# The launch script now automatically:
python scripts/dataset_tonic/setup_hf_dataset.py hf_your_token_here

# Creates: username/trackio-experiments
# Sets: TRACKIO_DATASET_REPO=username/trackio-experiments
```

### Custom Dataset Name

```bash
# Create with custom name
python scripts/dataset_tonic/setup_hf_dataset.py hf_your_token_here my-custom-experiments

# Creates: username/my-custom-experiments
# Sets: TRACKIO_DATASET_REPO=username/my-custom-experiments
```

### Launch Script Integration

The launch script now provides a seamless experience:

```bash
./launch.sh

# Step 3: Experiment Details
# - Automatically creates dataset repository
# - Option to use default or custom name
# - No manual username input required
```

## Features

### ✅ **Automatic Username Detection**
- Extracts username from HF token using Python API
- No manual username input required
- Consistent across all scripts

### ✅ **Flexible Dataset Naming**
- Default: `username/trackio-experiments`
- Custom: `username/custom-name`
- User choice during setup

### ✅ **Robust Error Handling**
- Graceful fallback to manual input
- Clear error messages
- Token validation before creation

### ✅ **Environment Integration**
- Automatically sets `TRACKIO_DATASET_REPO`
- Compatible with existing scripts
- No manual configuration required

### ✅ **Cross-Platform Compatibility**
- Works on Windows, Linux, macOS
- Uses Python API instead of CLI
- Consistent behavior across platforms

## Technical Implementation

### Token Authentication Flow

```python
# 1. Direct token authentication
api = HfApi(token=token)

# 2. Extract username
user_info = api.whoami()
username = user_info.get("name", user_info.get("username"))

# 3. Create repository
create_repo(
    repo_id=f"{username}/{dataset_name}",
    repo_type="dataset",
    token=token,
    exist_ok=True,
    private=False
)
```

### Launch Script Integration

```bash
# Automatic dataset creation
if python3 scripts/dataset_tonic/setup_hf_dataset.py 2>/dev/null; then
    TRACKIO_DATASET_REPO="$TRACKIO_DATASET_REPO"
    print_status "Dataset repository created successfully"
else
    # Fallback to manual input
    get_input "Trackio dataset repository" "$HF_USERNAME/trackio-experiments" TRACKIO_DATASET_REPO
fi
```

## User Experience Improvements

### Before (Manual Process)
1. User enters HF token
2. User manually types username
3. User manually types dataset repository name
4. User manually configures environment variables
5. Risk of typos and inconsistencies

### After (Automated Process)
1. User enters HF token
2. System automatically detects username
3. System automatically creates dataset repository
4. System automatically sets environment variables
5. Option to customize dataset name if desired

## Error Handling

### Common Scenarios

| Scenario | Action | User Experience |
|----------|--------|-----------------|
| Valid token | ✅ Automatic creation | Seamless setup |
| Invalid token | ❌ Clear error message | Helpful feedback |
| Network issues | ⚠️ Retry with fallback | Graceful degradation |
| Repository exists | ℹ️ Use existing | No conflicts |

### Fallback Mechanisms

1. **Token validation fails**: Clear error message with troubleshooting steps
2. **Dataset creation fails**: Fallback to manual input
3. **Network issues**: Retry with exponential backoff
4. **Permission issues**: Clear guidance on token permissions

## Benefits

### For Users
- **Simplified Setup**: No manual username input required
- **Reduced Errors**: Automatic username detection eliminates typos
- **Consistent Naming**: Standardized repository naming conventions
- **Better UX**: Clear progress indicators and feedback

### For Developers
- **Maintainable Code**: Python API instead of CLI dependencies
- **Cross-Platform**: Works consistently across operating systems
- **Extensible**: Easy to add new features and customizations
- **Testable**: Comprehensive test coverage

### For System
- **Reliable**: Robust error handling and fallback mechanisms
- **Secure**: Direct token authentication without environment conflicts
- **Scalable**: Easy to extend for additional repository types
- **Integrated**: Seamless integration with existing pipeline

## Migration Guide

### For Existing Users

No migration required! The system automatically:
- Detects existing repositories
- Uses existing repositories if they exist
- Creates new repositories only when needed

### For New Users

The setup is now completely automated:
1. Run `./launch.sh`
2. Enter your HF token
3. Choose dataset naming preference
4. System handles everything else automatically

## Future Enhancements

- [ ] Support for organization repositories
- [ ] Multiple dataset repositories per user
- [ ] Dataset repository templates
- [ ] Advanced repository configuration options
- [ ] Repository sharing and collaboration features

---

**Note**: This automation ensures that users can focus on their fine-tuning experiments rather than repository setup details, while maintaining full flexibility for customization when needed. 