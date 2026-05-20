# YOLO Training Dependency Conflict - Solution

## Problem Summary
The server has:
- PyTorch 2.6.0+cu124 (CUDA 12.4)
- opencv-python-headless 4.13.0.92
- ultralytics 8.4.51 installed

But ultralytics requires:
- ultralytics-thop (which needs torch 2.12.0 + CUDA 13.x) ❌ CONFLICT
- opencv-python>=4.6.0 (conflicts with opencv-python-headless) ❌ CONFLICT

## Root Cause
`ultralytics-thop` is used ONLY for model profiling (calculating FLOPs/params).
It's NOT required for actual YOLO training - only for `model.info()` calls.

## Solution Strategy

### Option 1: Skip ultralytics-thop (RECOMMENDED)
1. Don't install ultralytics-thop
2. Modify train_yolo.py to skip model.info() or catch the import error
3. Training will work fine, just won't show FLOPs/params stats

### Option 2: Use opencv-python-headless (Current Setup)
1. Keep opencv-python-headless (already installed)
2. Ultralytics should work with headless version for training
3. Only GUI visualization features will be unavailable (not needed on server)

### Option 3: Downgrade ultralytics
1. Try ultralytics 8.0.x or 8.1.x which might not require ultralytics-thop
2. Risk: May have different API or bugs

## Recommended Action

### Step 1: Verify current setup
```bash
cd /home/apulis-dev/userdata/lbh/danc/DanC
python -c "from ultralytics import YOLO; print('✓ ultralytics works')"
```

### Step 2: Test YOLO training without ultralytics-thop
```bash
# Try running train_yolo.py directly
python scripts/train_yolo.py
```

### Step 3: If it fails with thop import error
Modify `train_yolo.py` to catch the error:

```python
# Add at the top of train_yolo.py
import warnings
warnings.filterwarnings('ignore', message='.*thop.*')

# Wrap model.info() calls with try-except
try:
    model.info()
except Exception as e:
    print(f"Warning: Could not display model info: {e}")
    print("Training will continue normally...")
```

### Step 4: If opencv error occurs
The server has opencv-python-headless which should work for training.
If ultralytics complains, we can:
1. Ignore the warning (training should still work)
2. Or create a symlink: `opencv-python-headless` → `opencv-python`

## Expected Outcome
YOLO training should work WITHOUT installing ultralytics-thop.
The only missing feature will be FLOPs/params calculation, which is not critical.

## Files to Modify
- `scripts/train_yolo.py`: Add error handling for model.info()

## Testing Command
```bash
cd /home/apulis-dev/userdata/lbh/danc/DanC
bash scripts/train_all.sh  # Should now complete Step 2
```
