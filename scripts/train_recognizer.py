#!/usr/bin/env python3
"""
train_recognizer.py — Train ancient character recognition model (Stage 2).

Uses a ResNet50 backbone fine-tuned on cropped character images from training data.
Dataset is organized as: cropped_chars/{train,val}/{class_id}/*.png

Usage:
    python scripts/train_recognizer.py \
        --data_dir /home/apulis-dev/userdata/lbh/danc/cropped_chars \
        --output_dir /home/apulis-dev/userdata/lbh/danc/runs/recognize \
        --epochs 60 --batch 256 --gpus 0,1
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models


def build_transforms(img_size=64, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 8, img_size + 8)),
            transforms.RandomCrop(img_size),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def build_model(num_classes: int, backbone: str = "resnet50"):
    if backbone == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "resnet101":
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        if batch_idx % 100 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                  f"loss={loss.item():.4f} acc={100.*correct/total:.1f}%")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        _, top5_pred = outputs.topk(min(5, outputs.size(1)), dim=1)
        top5_correct += (top5_pred == labels.unsqueeze(1)).any(1).sum().item()

    return total_loss / total, correct / total, top5_correct / total


def main():
    parser = argparse.ArgumentParser(description="Train ancient char recognizer")
    parser.add_argument("--data_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/cropped_chars")
    parser.add_argument("--output_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/runs/recognize")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "resnet101", "efficientnet_b0"])
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size PER GPU")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    gpu_list = [int(g) for g in args.gpus.split(",")]
    use_ddp = len(gpu_list) > 1

    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(gpu_list[local_rank])
        device = torch.device(f"cuda:{gpu_list[local_rank]}")
        is_main = local_rank == 0
    else:
        device = torch.device(f"cuda:{gpu_list[0]}")
        is_main = True

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = data_dir / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    num_classes = meta["num_classes"]

    if is_main:
        print(f"Classes: {num_classes}")
        print(f"Device: {device} ({'DDP' if use_ddp else 'single GPU'})")

    train_transform = build_transforms(args.img_size, is_train=True)
    val_transform = build_transforms(args.img_size, is_train=False)

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_transform)

    if is_main:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"ImageFolder classes detected: {len(train_dataset.classes)}")

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch * 2, shuffle=False,
        sampler=val_sampler, num_workers=args.workers, pin_memory=True,
        persistent_workers=True,
    )

    actual_num_classes = len(train_dataset.classes)
    model = build_model(actual_num_classes, args.backbone).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if is_main:
            print(f"Resumed from: {args.resume}")

    if use_ddp:
        model = DDP(model, device_ids=[device.index])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0.0
    start_epoch = 0

    if args.resume and "epoch" in ckpt:
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)

    for epoch in range(start_epoch, args.epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss, val_acc, val_top5 = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        if is_main:
            print(f"Epoch {epoch}/{args.epochs-1} ({elapsed:.0f}s) "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top5={val_top5:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

            raw_model = model.module if use_ddp else model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_acc,
                "best_acc": best_acc,
                "num_classes": actual_num_classes,
                "backbone": args.backbone,
                "img_size": args.img_size,
                "class_to_idx": train_dataset.class_to_idx,
            }

            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint["best_acc"] = best_acc
                torch.save(checkpoint, output_dir / "best.pt")
                print(f"  -> New best: {best_acc:.4f}")

            if (epoch + 1) % 10 == 0:
                torch.save(checkpoint, output_dir / f"epoch_{epoch}.pt")

            torch.save(checkpoint, output_dir / "last.pt")

    if is_main:
        print(f"\nTraining complete. Best val acc: {best_acc:.4f}")
        print(f"Best model: {output_dir / 'best.pt'}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
