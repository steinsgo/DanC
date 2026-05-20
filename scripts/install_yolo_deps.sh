#!/bin/bash
# Workaround script to install ultralytics dependencies without ultralytics-thop
# This avoids the CUDA 13.x conflict

set -e

echo "=== Installing ultralytics dependencies (excluding ultralytics-thop) ==="

# Core dependencies that are missing
echo "Installing missing core dependencies..."
pip install --no-deps opencv-python>=4.6.0 || echo "opencv-python installation failed (might conflict with opencv-python-headless)"

# Try installing opencv-contrib-python instead (includes opencv-python functionality)
echo "Trying opencv-contrib-python as alternative..."
pip install --no-deps opencv-contrib-python>=4.6.0 || echo "opencv-contrib-python installation failed"

echo "=== Checking if YOLO training works without ultralytics-thop ==="
python scripts/check_thop.py

echo "=== Attempting dry-run of YOLO training ==="
python -c "
from ultralytics import YOLO
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
model = YOLO('yolov8n.yaml')
print('✓ YOLO model created successfully')
print('✓ Training should work without ultralytics-thop')
"
