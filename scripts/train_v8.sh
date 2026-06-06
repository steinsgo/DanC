#!/bin/bash
set -e

DANC=/home/apulis-dev/userdata/lbh/danc
LOGS=$DANC/logs
mkdir -p $LOGS

echo "=== v8: HUST-OBC pretrained + uniform (no Mixup) ===" | tee $LOGS/train_v8.log
echo "Start: $(date)" | tee -a $LOGS/train_v8.log

python /home/apulis-dev/userdata/lbh/danc/DanC/scripts/train_recognizer.py \
    --data_dir $DANC/cropped_chars \
    --output_dir $DANC/runs/recognize_v8 \
    --backbone resnet50 \
    --pretrained $DANC/runs/pretrain_hust/backbone.pth \
    --img_size 128 \
    --epochs 80 \
    --batch 192 \
    --lr 2e-4 \
    --workers 16 \
    --gpus 0 \
    --sampling uniform \
    --mixup_alpha 0 \
    2>&1 | tee -a $LOGS/train_v8.log

echo "=== Done: $(date) ===" | tee -a $LOGS/train_v8.log
