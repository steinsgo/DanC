#!/bin/bash
set -e

DANC=/home/apulis-dev/userdata/lbh/danc
LOGS=$DANC/logs
mkdir -p $LOGS

echo "=== Step 1: HUST-OBC pretrain ===" | tee $LOGS/train_v7.log
echo "Start: $(date)" | tee -a $LOGS/train_v7.log

python /home/apulis-dev/userdata/lbh/danc/DanC/scripts/pretrain_hust.py \
    --data_dir $DANC/HUST-OBC/deciphered \
    --output_dir $DANC/runs/pretrain_hust \
    --pretrained $DANC/DanC/models/resnet50_imagenet.pth \
    --epochs 30 --batch 192 --gpus 0 \
    2>&1 | tee -a $LOGS/train_v7.log

echo "=== Step 2: fine-tune on competition data ===" | tee -a $LOGS/train_v7.log
echo "Start: $(date)" | tee -a $LOGS/train_v7.log

python /home/apulis-dev/userdata/lbh/danc/DanC/scripts/train_recognizer.py \
    --data_dir $DANC/cropped_chars \
    --output_dir $DANC/runs/recognize_v7 \
    --backbone resnet50 \
    --pretrained $DANC/runs/pretrain_hust/backbone.pth \
    --img_size 128 \
    --epochs 60 \
    --batch 192 \
    --lr 2e-4 \
    --workers 16 \
    --gpus 0 \
    --sampling uniform \
    2>&1 | tee -a $LOGS/train_v7.log

echo "=== All done: $(date) ===" | tee -a $LOGS/train_v7.log
