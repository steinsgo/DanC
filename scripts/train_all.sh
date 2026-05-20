#!/bin/bash
set -e

# =============================================================================
# train_all.sh — Run all training steps in sequence
#
# Usage:
#   bash scripts/train_all.sh          # Full pipeline
#   bash scripts/train_all.sh --step 2 # Start from step 2
# =============================================================================

BASE="/home/apulis-dev/userdata/lbh/danc"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STEP=${1:-1}

if [ "$1" = "--step" ]; then
    STEP=$2
fi

echo "=============================================="
echo " Ancient OCR Training Pipeline"
echo " Base dir: $BASE"
echo " Starting from step: $STEP"
echo "=============================================="

# ------------------------------------------------------------------
# Step 1: Crop characters from training images (CPU, ~10-20 min)
# ------------------------------------------------------------------
if [ "$STEP" -le 1 ]; then
    echo ""
    echo "[Step 1/4] Cropping training characters..."
    python "$SCRIPT_DIR/crop_train_chars.py" \
        --train_dir "$BASE/train/out_of_domain" \
        --output_dir "$BASE/cropped_chars" \
        --char_dict "$BASE/char_dict.json" \
        --val_ratio 0.15 --seed 42
    echo "[Step 1/4] Done."
fi

# ------------------------------------------------------------------
# Step 2: Train YOLO detector (GPU 0+1, ~4-8 hours)
# ------------------------------------------------------------------
if [ "$STEP" -le 2 ]; then
    echo ""
    echo "[Step 2/4] Training YOLO detector..."

    # Try pretrained weights, fall back to from-scratch
    WEIGHTS="$BASE/DanC/model/yolo11m.pt"
    WEIGHTS_ARG=""
    if [ -f "$WEIGHTS" ]; then
        WEIGHTS_ARG="--weights $WEIGHTS"
        echo "  Using pretrained: $WEIGHTS"
    else
        WEIGHTS_ARG="--weights none --model_cfg yolo11m.yaml"
        echo "  No pretrained weights found, training from scratch"
    fi

    python "$SCRIPT_DIR/train_yolo.py" \
        --data "$BASE/yolo_dataset/dataset.yaml" \
        $WEIGHTS_ARG \
        --epochs 150 \
        --imgsz 1280 \
        --batch 16 \
        --device 0 \
        --project "$BASE/runs/detect" \
        --name ancient_char_det

    echo "[Step 2/4] Done. Best model: $BASE/runs/detect/ancient_char_det/weights/best.pt"
fi

# ------------------------------------------------------------------
# Step 3: Train character recognizer (GPU 0+1, ~2-4 hours)
# ------------------------------------------------------------------
if [ "$STEP" -le 3 ]; then
    echo ""
    echo "[Step 3/4] Training character recognizer..."
    torchrun --nproc_per_node=2 "$SCRIPT_DIR/train_recognizer.py" \
        --data_dir "$BASE/cropped_chars" \
        --output_dir "$BASE/runs/recognize" \
        --backbone resnet50 \
        --img_size 64 \
        --epochs 60 \
        --batch 256 \
        --lr 1e-3 \
        --gpus 0,1
    echo "[Step 3/4] Done. Best model: $BASE/runs/recognize/best.pt"
fi

# ------------------------------------------------------------------
# Step 4: Quick sanity check inference on a few training images
# ------------------------------------------------------------------
if [ "$STEP" -le 4 ]; then
    echo ""
    echo "[Step 4/4] Sanity check inference..."

    # Create a small test set from training images
    TEST_DIR="$BASE/sanity_test"
    mkdir -p "$TEST_DIR"
    ls "$BASE/train/out_of_domain/"*.png | head -5 | while read f; do
        ln -sf "$f" "$TEST_DIR/"
    done

    python "$SCRIPT_DIR/run_inference.py" \
        --input_dir "$TEST_DIR" \
        --det_model "$BASE/runs/detect/ancient_char_det/weights/best.pt" \
        --rec_model "$BASE/runs/recognize/best.pt" \
        --char_dict "$BASE/char_dict.json" \
        --output "$BASE/sanity_test/prediction.json" \
        --device cuda:0

    echo ""
    echo "Sanity check results:"
    python -c "
import json
with open('$BASE/sanity_test/prediction.json') as f:
    d = json.load(f)
for k, v in d.items():
    print(f'  {k}: {len(v)} chars detected')
    for c in v[:5]:
        print(f'    bbox={c[\"bbox\"]} text={c[\"text\"]}')
"
    echo "[Step 4/4] Done."
fi

echo ""
echo "=============================================="
echo " Training pipeline complete!"
echo " Detection model: $BASE/runs/detect/ancient_char_det/weights/best.pt"
echo " Recognition model: $BASE/runs/recognize/best.pt"
echo " Char dict: $BASE/char_dict.json"
echo "=============================================="
