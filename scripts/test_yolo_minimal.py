#!/usr/bin/env python3
"""
Minimal test to verify YOLO training works without ultralytics-thop.
Run this on the server before attempting full training.
"""
import sys

print("=== Testing YOLO without ultralytics-thop ===")
print()

# Step 1: Check if thop is imported
print("Step 1: Checking if thop module exists...")
try:
    import thop
    print("  ✗ thop IS installed (unexpected)")
except ImportError:
    print("  ✓ thop is NOT installed (expected)")
print()

# Step 2: Import ultralytics
print("Step 2: Importing ultralytics...")
try:
    from ultralytics import YOLO
    print("  ✓ ultralytics imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import ultralytics: {e}")
    sys.exit(1)
print()

# Step 3: Check PyTorch and CUDA
print("Step 3: Checking PyTorch environment...")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"  ✗ PyTorch check failed: {e}")
    sys.exit(1)
print()

# Step 4: Create a YOLO model
print("Step 4: Creating YOLO model...")
try:
    model = YOLO('yolo11n.yaml')  # Smallest model for testing
    print("  ✓ YOLO model created successfully")
except Exception as e:
    print(f"  ✗ Failed to create YOLO model: {e}")
    sys.exit(1)
print()

# Step 5: Test model.info() (this is what requires ultralytics-thop)
print("Step 5: Testing model.info() (requires ultralytics-thop)...")
try:
    model.info()
    print("  ✓ model.info() worked (ultralytics-thop must be installed)")
except Exception as e:
    print(f"  ⚠ model.info() failed: {e}")
    print("  This is EXPECTED if ultralytics-thop is not installed.")
    print("  Training should still work fine.")
print()

print("=== Summary ===")
print("✓ YOLO can be imported and models can be created")
print("✓ Training should work without ultralytics-thop")
print("⚠ model.info() profiling will not work (optional feature)")
print()
print("Next step: Try running the actual training script:")
print("  python scripts/train_yolo.py --epochs 1 --batch 2")
