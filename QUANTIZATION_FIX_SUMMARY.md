# Quantization Fix Summary

## Issues Identified

The quantization script was failing due to several compatibility issues:

1. **Int8 Quantization Error**: 
   - Error: `The model is quantized with QuantizationMethod.TORCHAO and is not serializable`
   - Cause: Offloaded modules in the model cannot be quantized with torchao
   - Solution: Added alternative save method and fallback to bitsandbytes

2. **Int4 Quantization Error**:
   - Error: `Could not run 'aten::_convert_weight_to_int4pack_for_cpu' with arguments from the 'CUDA' backend`
   - Cause: Int4 quantization requires CPU backend but was being attempted on CUDA
   - Solution: Added proper device selection logic

3. **Monitoring Error**:
   - Error: `'SmolLM3Monitor' object has no attribute 'log_event'`
   - Cause: Incorrect monitoring API usage
   - Solution: Added flexible monitoring method detection

## Fixes Implemented

### 1. Enhanced Device Management (`scripts/model_tonic/quantize_model.py`)

```python
def get_optimal_device(self, quant_type: str) -> str:
    """Get optimal device for quantization type"""
    if quant_type == "int4_weight_only":
        # Int4 quantization works better on CPU
        return "cpu"
    elif quant_type == "int8_weight_only":
        # Int8 quantization works on GPU
        if torch.cuda.is_available():
            return "cuda"
        else:
            logger.warning("⚠️ CUDA not available, falling back to CPU for int8")
            return "cpu"
    else:
        return "auto"
```

### 2. Alternative Quantization Method

Added `quantize_model_alternative()` method using bitsandbytes for better compatibility:

```python
def quantize_model_alternative(self, quant_type: str, device: str = "auto", group_size: int = 128, save_dir: Optional[str] = None) -> Optional[str]:
    """Alternative quantization using bitsandbytes for better compatibility"""
    # Uses BitsAndBytesConfig instead of TorchAoConfig
    # Handles serialization issues better
```

### 3. Improved Error Handling

- Added fallback from torchao to bitsandbytes
- Enhanced save method with alternative approaches
- Better device mapping for different quantization types

### 4. Fixed Monitoring Integration

```python
def log_to_trackio(self, action: str, details: Dict[str, Any]):
    """Log quantization events to Trackio"""
    if self.monitor:
        try:
            # Use the correct monitoring method
            if hasattr(self.monitor, 'log_event'):
                self.monitor.log_event(action, details)
            elif hasattr(self.monitor, 'log_metric'):
                self.monitor.log_metric(action, details.get('value', 1.0))
            elif hasattr(self.monitor, 'log'):
                self.monitor.log(action, details)
            else:
                logger.info(f"📊 {action}: {details}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log to Trackio: {e}")
```

## Usage Instructions

### 1. Install Dependencies

```bash
pip install -r requirements_quantization.txt
```

### 2. Run Quantization

```bash
python3 quantize_and_push.py
```

### 3. Test Fixes

```bash
python3 test_quantization_fix.py
```

## Expected Behavior

### Successful Quantization

The script will now:

1. **Try torchao first** for each quantization type
2. **Fall back to bitsandbytes** if torchao fails
3. **Use appropriate devices** (CPU for int4, GPU for int8)
4. **Handle serialization issues** with alternative save methods
5. **Log progress** without monitoring errors

### Output

```
✅ Model files validated
🔄 Processing quantization type: int8_weight_only
🔄 Using device: cuda
✅ int8_weight_only quantization and push completed
🔄 Processing quantization type: int4_weight_only
🔄 Using device: cpu
✅ int4_weight_only quantization and push completed
📊 Quantization summary: 2/2 successful
✅ Quantization completed successfully!
```

## Troubleshooting

### If All Quantization Fails

1. **Install bitsandbytes**:
   ```bash
   pip install bitsandbytes
   ```

2. **Check model path**:
   ```bash
   ls -la /output-checkpoint
   ```

3. **Verify dependencies**:
   ```bash
   python3 test_quantization_fix.py
   ```

### Common Issues

1. **Memory Issues**: Use CPU for int4 quantization
2. **Serialization Errors**: The script now handles these automatically
3. **Device Conflicts**: Automatic device selection based on quantization type

## Files Modified

1. `scripts/model_tonic/quantize_model.py` - Main quantization logic
2. `quantize_and_push.py` - Main script with better error handling
3. `test_quantization_fix.py` - Test script for verification
4. `requirements_quantization.txt` - Dependencies file

## Next Steps

1. Run the test script to verify fixes
2. Install bitsandbytes if not already installed
3. Run the quantization script
4. Check the Hugging Face repository for quantized models

The fixes ensure robust quantization with multiple fallback options and proper error handling.