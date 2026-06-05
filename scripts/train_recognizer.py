#!/usr/bin/env python3
"""
train_recognizer.py — Train ancient character recognition model (Stage 2).

Uses a ResNet50 backbone fine-tuned on cropped character images from training data.
Dataset is organized as: cropped_chars/{train,val}/{class_id}/*.png

Long-tail rebalance:
    --sampling uniform  : standard ImageFolder sampling (default before)
    --sampling sqrt     : per-sample weight = 1/sqrt(class_count)  (recommended)
    --sampling balanced : per-sample weight = 1/class_count

Usage (single GPU):
    python scripts/train_recognizer.py --gpus 0 --sampling sqrt

Usage (DDP):
    torchrun --nproc_per_node=2 scripts/train_recognizer.py --gpus 0,1 --sampling sqrt
"""
import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from torchvision import datasets, transforms, models


# ---------------------------------------------------------------------------
# Custom samplers
# ---------------------------------------------------------------------------

class DistributedWeightedSampler(Sampler):
    """
    Weighted sampling that's also DDP-aware.

    Each epoch every rank seeds the same generator (seed + epoch), draws the
    same global index pool with multinomial(weights, num_samples, replacement=True),
    then takes its own slice. This guarantees no overlap between ranks and
    consistent behavior across runs.
    """
    def __init__(self, weights, num_samples, num_replicas=None, rank=None, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        # Round up so num_samples is divisible by num_replicas
        self.num_samples_per_rank = math.ceil(num_samples / num_replicas)
        self.total_samples = self.num_samples_per_rank * num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        all_indices = torch.multinomial(
            self.weights, self.total_samples, replacement=True, generator=g
        )
        start = self.rank * self.num_samples_per_rank
        end = start + self.num_samples_per_rank
        return iter(all_indices[start:end].tolist())

    def __len__(self):
        return self.num_samples_per_rank

    def set_epoch(self, epoch):
        self.epoch = epoch


def compute_sample_weights(targets, num_classes, strategy="sqrt"):
    """
    Compute per-sample weights for long-tail rebalancing.

    Returns:
        weights: np.ndarray of length len(targets), each entry is the
                 sampling weight for that sample.
        class_counts: dict {class_id: count} for diagnostics.
    """
    counter = Counter(targets)
    class_counts = np.array([counter.get(c, 0) for c in range(num_classes)],
                            dtype=np.float64)
    # Avoid div-by-zero for unseen classes
    safe_counts = np.where(class_counts > 0, class_counts, 1.0)

    if strategy == "sqrt":
        per_class_weight = 1.0 / np.sqrt(safe_counts)
    elif strategy == "balanced":
        per_class_weight = 1.0 / safe_counts
    elif strategy == "uniform":
        per_class_weight = np.ones_like(safe_counts)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    # Normalize so the maximum class weight is 1 (numerical stability)
    per_class_weight = per_class_weight / per_class_weight.max()

    weights = np.array([per_class_weight[t] for t in targets], dtype=np.float64)
    return weights, dict(counter)


def bucket_classes(class_counts, num_classes):
    """
    Split classes into head/middle/tail buckets by frequency.

    Returns: (head_set, mid_set, tail_set) of class indices.
    """
    counts = np.array([class_counts.get(c, 0) for c in range(num_classes)])
    sorted_classes = np.argsort(-counts)  # most frequent first

    n = num_classes
    head = set(sorted_classes[: n // 3].tolist())
    mid = set(sorted_classes[n // 3: 2 * n // 3].tolist())
    tail = set(sorted_classes[2 * n // 3:].tolist())
    return head, mid, tail


# ---------------------------------------------------------------------------
# Transforms / model
# ---------------------------------------------------------------------------

def build_transforms(img_size=128, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def build_model(num_classes: int, backbone: str = "resnet50", pretrained_path: str = ""):
    if backbone == "resnet50":
        model = models.resnet50(weights=None)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            print(f"Loaded pretrained weights: {pretrained_path}")
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes),
        )
    elif backbone == "resnet101":
        model = models.resnet101(weights=None)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            print(f"Loaded pretrained weights: {pretrained_path}")
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes),
        )
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

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
def validate(model, loader, criterion, device, head_set=None, mid_set=None, tail_set=None):
    """Validate. If buckets provided, also report per-bucket accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0

    bucket_stats = {"head": [0, 0], "mid": [0, 0], "tail": [0, 0]}

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        match = predicted.eq(labels)
        correct += match.sum().item()
        total += images.size(0)

        _, top5_pred = outputs.topk(min(5, outputs.size(1)), dim=1)
        top5_correct += (top5_pred == labels.unsqueeze(1)).any(1).sum().item()

        if head_set is not None:
            labels_cpu = labels.cpu().numpy()
            match_cpu = match.cpu().numpy()
            for lbl, m in zip(labels_cpu, match_cpu):
                lbl = int(lbl)
                if lbl in head_set:
                    bucket = "head"
                elif lbl in mid_set:
                    bucket = "mid"
                else:
                    bucket = "tail"
                bucket_stats[bucket][0] += int(m)
                bucket_stats[bucket][1] += 1

    bucket_accs = {}
    for k, (c, t) in bucket_stats.items():
        bucket_accs[k] = (c / t) if t > 0 else 0.0

    return total_loss / total, correct / total, top5_correct / total, bucket_accs


def main():
    parser = argparse.ArgumentParser(description="Train ancient char recognizer")
    parser.add_argument("--data_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/cropped_chars")
    parser.add_argument("--output_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/runs/recognize")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "resnet101", "efficientnet_b0"])
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128, help="Batch size PER GPU")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--pretrained", type=str, default="",
                        help="Path to pretrained backbone weights")
    parser.add_argument("--sampling", type=str, default="sqrt",
                        choices=["uniform", "sqrt", "balanced"],
                        help="Long-tail rebalance sampling: sqrt=1/sqrt(count), balanced=1/count")
    parser.add_argument("--samples_per_epoch_mult", type=float, default=1.0,
                        help="Multiplier for # of samples per epoch (only when sampling != uniform)")
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
        print(f"Classes (from meta): {num_classes}")
        print(f"Device: {device} ({'DDP' if use_ddp else 'single GPU'})")
        print(f"Sampling strategy: {args.sampling}")

    train_transform = build_transforms(args.img_size, is_train=True)
    val_transform = build_transforms(args.img_size, is_train=False)

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_transform)

    # Force val to use the same class_to_idx as train
    val_dataset.class_to_idx = train_dataset.class_to_idx
    val_dataset.classes = train_dataset.classes
    val_samples = []
    for path, _ in val_dataset.samples:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name in train_dataset.class_to_idx:
            val_samples.append((path, train_dataset.class_to_idx[class_name]))
    val_dataset.samples = val_samples
    val_dataset.targets = [s[1] for s in val_samples]

    actual_num_classes = len(train_dataset.classes)

    if is_main:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples:   {len(val_dataset)}")
        print(f"Actual classes: {actual_num_classes}")

    # ---- Build train sampler (uniform / sqrt / balanced) ----
    train_targets = train_dataset.targets
    sample_weights, train_class_counts = compute_sample_weights(
        train_targets, actual_num_classes, strategy=args.sampling
    )

    if is_main:
        counts_arr = np.array(list(train_class_counts.values()))
        print(f"Train class freq: min={counts_arr.min()} median={int(np.median(counts_arr))} "
              f"max={counts_arr.max()} mean={counts_arr.mean():.1f}")
        n_singleton = int((counts_arr == 1).sum())
        n_under5 = int((counts_arr < 5).sum())
        print(f"Long-tail: classes with count=1: {n_singleton}, count<5: {n_under5}")

    head_set, mid_set, tail_set = bucket_classes(train_class_counts, actual_num_classes)

    if args.sampling == "uniform":
        if use_ddp:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            train_sampler = None
    else:
        num_samples = int(len(train_dataset) * args.samples_per_epoch_mult)
        if use_ddp:
            train_sampler = DistributedWeightedSampler(
                weights=sample_weights, num_samples=num_samples,
                num_replicas=dist.get_world_size(), rank=dist.get_rank(), seed=42,
            )
        else:
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights, num_samples=num_samples, replacement=True,
            )

    if use_ddp:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch,
        shuffle=(train_sampler is None and args.sampling == "uniform"),
        sampler=train_sampler, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch * 2, shuffle=False,
        sampler=val_sampler, num_workers=args.workers, pin_memory=True,
        persistent_workers=True,
    )

    model = build_model(actual_num_classes, args.backbone, args.pretrained).to(device)

    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if is_main:
            print(f"Resumed from: {args.resume}")

    if use_ddp:
        model = DDP(model, device_ids=[device.index])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0.0
    start_epoch = 0
    if ckpt and "epoch" in ckpt:
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)

    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss, val_acc, val_top5, bucket_accs = validate(
            model, val_loader, criterion, device,
            head_set=head_set, mid_set=mid_set, tail_set=tail_set,
        )
        scheduler.step()

        elapsed = time.time() - t0

        if is_main:
            print(f"Epoch {epoch}/{args.epochs-1} ({elapsed:.0f}s) "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top5={val_top5:.4f} "
                  f"head={bucket_accs['head']:.3f} mid={bucket_accs['mid']:.3f} "
                  f"tail={bucket_accs['tail']:.3f} "
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
