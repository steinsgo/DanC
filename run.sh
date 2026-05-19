#!/bin/bash
set -e

echo "=== Ancient Character OCR Inference ==="
echo "Start time: $(date)"

[ -d "/saisdata" ] || { echo "Error: /saisdata not found"; exit 1; }
[ -d "/saisresult" ] || { echo "Error: /saisresult not found"; exit 1; }

INPUT_DIR="/saisdata/13/eval/images"
if [ ! -d "$INPUT_DIR" ]; then
    INPUT_DIR="/saisdata"
    echo "Warning: /saisdata/13/eval/images not found, using /saisdata"
fi

echo "Input: $INPUT_DIR"
echo "Output: /saisresult/prediction.json"

python /app/src/run_inference.py \
    --input_dir "$INPUT_DIR" \
    --det_model /app/models/best_det.pt \
    --rec_model /app/models/best_rec.pt \
    --char_dict /app/models/char_dict.json \
    --output /saisresult/prediction.json \
    --device cuda:0

[ -f "/saisresult/prediction.json" ] || { echo "Error: prediction.json not generated"; exit 1; }

echo "Done! $(date)"
