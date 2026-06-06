#!/usr/bin/env python3
"""
train_recognizer.py — Train ancient character recognition model (Stage 2).

Improvements over v5:
  - img_size default: 128 → 160
  - EMA (Exponential Moving Average) of model weights
  - Mixup augmentation for head/mid classes only

Long-tail rebalance:
    --sampling uniform  : standard ImageFolder sampling
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
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from torchvision import datasets, transforms, models


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = (model.module if hasattr(model, "module") else model).state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(self.decay).add_((1 - self.decay) * msd[k].detach())


# ---------------------------------------------------------------------------
# Custom samplers
# ---------------------------------------------------------------------------

class DistributedWeightedSampler(Sampler):
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
        self.num_samples_per_rank = math.ceil(num_samples / num_replicas)
        self.total_samples = self.num_samples_per_rank * num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        all_indices = torch.multinomial(
            self.weights, self.total_samples, replacement=True, generator=g
        )
        start = self.rank * self.num_samples_per_rank
        return iter(all_indices[start:start + self.num_samples_per_rank].tolist())

    def __len__(self):
        return self.num_samples_per_rank

    def set_epoch(self, epoch):
        self.epoch = epoch


def compute_sample_weights(targets, num_classes, strategy="sqrt"):
    counter = Counter(targets)
    class_counts = np.array([counter.get(c, 0) for c in range(num_classes)], dtype=np.float64)
    safe_counts = np.where(class_counts > 0, class_counts, 1.0)
    if strategy == "sqrt":
        per_class_weight = 1.0 / np.sqrt(safe_counts)
    elif strategy == "balanced":
        per_class_weight = 1.0 / safe_counts
    elif strategy == "uniform":
        per_class_weight = np.ones_like(safe_counts)
    elif strategy.startswith("power"):
        power = float(strategy.replace("power", ""))
        per_class_weight = 1.0 / np.power(safe_counts, power)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    per_class_weight = per_class_weight / per_class_weight.max()
    weights = np.array([per_class_weight[t] for t in targets], dtype=np.float64)
    return weights, dict(counter)


def bucket_classes(class_counts, num_classes):
    counts = np.array([class_counts.get(c, 0) for c in range(num_classes)])
    sorted_classes = np.argsort(-counts)
    n = num_classes
    head = set(sorted_classes[:n // 3].tolist())
    mid = set(sorted_classes[n // 3: 2 * n // 3].tolist())
    tail = set(sorted_classes[2 * n // 3:].tolist())
    return head, mid, tail


# ---------------------------------------------------------------------------
# Transforms / model
# ---------------------------------------------------------------------------

def build_transforms(img_size=160, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.RandomCrop(img_size),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def build_model(num_classes: int, backbone: str = "resnet50", pretrained_path: str = ""):
    if backbone == "resnet50":
        model = models.resnet50(weights=None)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights: {pretrained_path} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    elif backbone == "resnet101":
        model = models.resnet101(weights=None)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights: {pretrained_path} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def mixup_batch(images, labels, head_set, mid_set, alpha=0.4):
    """Apply Mixup only to samples whose label is in head or mid classes."""
    device = images.device
    if alpha <= 0:
        return images, labels, labels, torch.ones(len(labels), device=device)
    mask = torch.tensor(
        [int(l.item()) in head_set or int(l.item()) in mid_set for l in labels],
        dtype=torch.bool, device=device,
    )
    if mask.sum() < 2:
        return images, labels, labels, torch.ones(len(labels), device=device)

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(len(images), device=device)
    mixed = images.clone()
    mixed[mask] = lam * images[mask] + (1 - lam) * images[idx][mask]
    return mixed, labels, labels[idx], torch.where(mask, torch.full_like(mask, lam, dtype=torch.float), torch.ones(len(labels), device=device, dtype=torch.float))


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    ema, head_set, mid_set, mixup_alpha=0.4):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images, labels_a, labels_b, lam = mixup_batch(images, labels, head_set, mid_set, alpha=mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = (lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

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
                bucket = "head" if lbl in head_set else ("mid" if lbl in mid_set else "tail")
                bucket_stats[bucket][0] += int(m)
                bucket_stats[bucket][1] += 1

    bucket_accs = {k: (c / t) if t > 0 else 0.0 for k, (c, t) in bucket_stats.items()}
    return total_loss / total, correct / total, top5_correct / total, bucket_accs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ancient char recognizer")
    parser.add_argument("--data_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/cropped_chars")
    parser.add_argument("--output_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/runs/recognize")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "resnet101", "efficientnet_b0"])
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--sampling", type=str, default="sqrt")
    parser.add_argument("--samples_per_epoch_mult", type=float, default=1.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.4,
                        help="Mixup alpha; set 0 to disable")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
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

    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    num_classes = meta["num_classes"]

    if is_main:
        print(f"Classes (from meta): {num_classes}")
        print(f"Device: {device} ({'DDP' if use_ddp else 'single GPU'})")
        print(f"Sampling: {args.sampling}  img_size: {args.img_size}  "
              f"mixup_alpha: {args.mixup_alpha}  ema_decay: {args.ema_decay}")

    train_dataset = datasets.ImageFolder(data_dir / "train",
                                         transform=build_transforms(args.img_size, True))
    val_dataset = datasets.ImageFolder(data_dir / "val",
                                       transform=build_transforms(args.img_size, False))

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

    sample_weights, train_class_counts = compute_sample_weights(
        train_dataset.targets, actual_num_classes, strategy=args.sampling
    )
    head_set, mid_set, tail_set = bucket_classes(train_class_counts, actual_num_classes)

    if is_main:
        counts_arr = np.array(list(train_class_counts.values()))
        print(f"Train: {len(train_dataset)} samples, {actual_num_classes} classes | "
              f"Val: {len(val_dataset)}")
        print(f"Class freq: min={counts_arr.min()} median={int(np.median(counts_arr))} "
              f"max={counts_arr.max()}")

    if args.sampling == "uniform":
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
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

    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None

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
    ema = ModelEMA(model, decay=args.ema_decay)

    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "ema_state_dict" in ckpt:
            ema.ema.load_state_dict(ckpt["ema_state_dict"])
        if is_main:
            print(f"Resumed from: {args.resume}")

    if use_ddp:
        model = DDP(model, device_ids=[device.index])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2, reduction="none")
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
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            ema, head_set, mid_set, args.mixup_alpha,
        )
        # validate with EMA weights
        val_loss, val_acc, val_top5, bucket_accs = validate(
            ema.ema, val_loader, nn.CrossEntropyLoss(label_smoothing=0.2),
            device, head_set=head_set, mid_set=mid_set, tail_set=tail_set,
        )
        scheduler.step()

        if is_main:
            print(f"Epoch {epoch}/{args.epochs-1} ({time.time()-t0:.0f}s) "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top5={val_top5:.4f} "
                  f"head={bucket_accs['head']:.3f} mid={bucket_accs['mid']:.3f} "
                  f"tail={bucket_accs['tail']:.3f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

            raw_model = model.module if use_ddp else model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "ema_state_dict": ema.ema.state_dict(),
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
