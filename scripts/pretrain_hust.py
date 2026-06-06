#!/usr/bin/env python3
"""
pretrain_hust.py — 用 HUST-OBC deciphered 数据预训练 ResNet backbone。

数据目录结构: data_dir/{class_id}/*.png  (无 train/val 子目录)
自动按 9:1 切分 train/val，跑完保存纯 backbone 权重到 output_dir/backbone.pth。

Usage:
    python scripts/pretrain_hust.py \
        --data_dir /home/apulis-dev/userdata/lbh/danc/HUST-OBC/deciphered \
        --output_dir /home/apulis-dev/userdata/lbh/danc/runs/pretrain_hust \
        --pretrained /home/apulis-dev/userdata/lbh/danc/DanC/models/resnet50_imagenet.pth \
        --epochs 30 --batch 192 --gpus 0
"""
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def build_transforms(img_size, is_train):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/HUST-OBC/deciphered")
    parser.add_argument("--output_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/runs/pretrain_hust")
    parser.add_argument("--pretrained", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/DanC/models/resnet50_imagenet.pth")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpus.split(',')[0]}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 用 train transform 加载全量，之后切 subset
    full_dataset = datasets.ImageFolder(args.data_dir,
                                        transform=build_transforms(args.img_size, True))
    num_classes = len(full_dataset.classes)
    n = len(full_dataset)

    # 按类别做 9:1 切分，保证每类都有 val 样本
    train_indices, val_indices = [], []
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)
    for label, idxs in class_indices.items():
        split = max(1, len(idxs) // 10)
        val_indices.extend(idxs[:split])
        train_indices.extend(idxs[split:])

    val_dataset = datasets.ImageFolder(args.data_dir,
                                       transform=build_transforms(args.img_size, False))
    train_loader = DataLoader(
        Subset(full_dataset, train_indices), batch_size=args.batch,
        shuffle=True, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices), batch_size=args.batch * 2,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        persistent_workers=True,
    )

    print(f"Classes: {num_classes}  Train: {len(train_indices)}  Val: {len(val_indices)}")

    model = models.resnet50(weights=None)
    if args.pretrained:
        model.load_state_dict(
            torch.load(args.pretrained, map_location="cpu", weights_only=True), strict=True
        )
        print(f"Loaded: {args.pretrained}")
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # validate
        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast("cuda"):
                    vc += (model(images).argmax(1) == labels).sum().item()
                vt += labels.size(0)
        val_acc = vc / vt

        print(f"Epoch {epoch}/{args.epochs-1} ({time.time()-t0:.0f}s) "
              f"val_acc={val_acc:.4f} lr={scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_full.pt")
            print(f"  -> New best: {best_acc:.4f}")

    # 保存纯 backbone 权重（去掉 fc）供 fine-tune 使用
    backbone_state = {k: v for k, v in model.state_dict().items()
                      if not k.startswith("fc.")}
    torch.save(backbone_state, output_dir / "backbone.pth")
    print(f"\nDone. Best val acc: {best_acc:.4f}")
    print(f"Backbone saved: {output_dir / 'backbone.pth'}")


if __name__ == "__main__":
    main()
