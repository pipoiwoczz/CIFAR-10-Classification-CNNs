"""
Train all three models (SimpleCNN, ResNet-18 for CIFAR, WideResNet-28-10)
using the existing CV pipeline in this repo. Also supports a final
single-split train to produce a single best checkpoint per model.

Usage (cross-validation only):
    python train_all_models.py --cv

Usage (final single split; 90/10 val):
    python train_all_models.py --final --epochs 120 --batch-size 256

This script relies on modules in the same directory:
    - config.py, model.py, data.py, cv.py, utils.py
"""
from __future__ import annotations
import os
import json
import argparse
from dataclasses import replace

import torch
import torch.nn as nn

from src.config import CFG
from src.model import build_model
from src.data import load_full_train_without_transform, make_fold_loaders
from src.utils import EarlyStopping, set_seed, cosine_with_warmup_lambda
from src.cv import evaluate

from sklearn.model_selection import StratifiedShuffleSplit


# ----------------------------- Helpers ---------------------------------

def make_single_split_loaders(base_train, cfg: CFG, val_frac: float = 0.1):
    """Create a single 90/10 stratified split with repo-standard transforms."""
    import numpy as np
    y_all = np.array(base_train.targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=cfg.seed)
    (train_idx, val_idx), = sss.split(X=np.zeros_like(y_all), y=y_all)
    train_loader, val_loader = make_fold_loaders(base_train, train_idx, val_idx, cfg)
    return train_loader, val_loader


def train_one_model(cfg: CFG, model_name: str, train_loader, val_loader, outdir: str) -> dict:
    """Generic training loop shared across all three architectures.

    Returns a dict with best metrics and the path to the best checkpoint.
    """
    os.makedirs(outdir, exist_ok=True)

    device = torch.device(cfg.device)
    model = build_model(model_name, num_classes=10, dropout=cfg.dropout).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.base_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    lr_lambda = lambda e: cosine_with_warmup_lambda(e, cfg.epochs, cfg.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and cfg.use_amp))
    early = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    best = {"val_acc": 0.0, "val_loss": float("inf"), "epoch": 0}
    best_ckpt = os.path.join(outdir, f"{model_name}_best.pt")

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and cfg.use_amp)):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        scheduler.step()

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_loss = running_loss / max(1, seen)
        print(
            f"[{model_name}] Epoch {epoch+1:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        # Early-stopping check uses accuracy as the monitored metric
        if early.step(val_acc):
            best.update({"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch + 1})
            torch.save(model.state_dict(), best_ckpt)

        if early.stop:
            print(f"[{model_name}] Early stopping at epoch {epoch+1}. Best val_acc={best['val_acc']:.4f}")
            break

    best["checkpoint"] = best_ckpt
    return best


# ----------------------------- Main runners -----------------------------

def run_cross_validation(cfg: CFG, model_names: list[str]) -> None:
    """Loop models and use the repo's fold loader builder for each fold.

    This mimics the logic in cv.py but executes per-model and aggregates results
    to a single JSON per model.
    """
    import numpy as np
    import torchvision as tv
    from sklearn.model_selection import StratifiedKFold

    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    base_train = tv.datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=None)
    y_all = np.array(base_train.targets)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle_folds, random_state=cfg.seed)

    for name in model_names:
        fold_accs = []
        print(f"===== Cross-validation for {name} =====")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros_like(y_all), y=y_all), start=1):
            fold_dir = os.path.join(cfg.save_dir, f"{name}_fold{fold}")
            train_loader, val_loader = make_fold_loaders(base_train, train_idx, val_idx, cfg)
            best = train_one_model(cfg, name, train_loader, val_loader, fold_dir)
            fold_accs.append(best["val_acc"])

        summary = {
            "model": name,
            "val_mean": float(np.mean(fold_accs)),
            "val_std": float(np.std(fold_accs)),
            "fold_accs": [float(a) for a in fold_accs],
        }
        out_json = os.path.join(cfg.save_dir, f"{name}_cv_summary.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved CV summary to: {out_json}\n{summary}\n")


def run_final_single_split(cfg: CFG, model_names: list[str]) -> None:
    """Train each model on a single 90/10 split and save the best checkpoint."""
    import torchvision as tv

    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    base_train = load_full_train_without_transform(cfg.data_root)
    train_loader, val_loader = make_single_split_loaders(base_train, cfg, val_frac=0.1)

    for name in model_names:
        outdir = os.path.join(cfg.save_dir, f"{name}_final")
        best = train_one_model(cfg, name, train_loader, val_loader, outdir)
        # Optionally evaluate the best checkpoint again
        device = torch.device(cfg.device)
        model = build_model(name, num_classes=10, dropout=cfg.dropout).to(device)
        model.load_state_dict(torch.load(best["checkpoint"], map_location=device))
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        best.update({"val_acc_reloaded": float(val_acc), "val_loss_reloaded": float(val_loss)})

        with open(os.path.join(outdir, f"{name}_final_summary.json"), "w") as f:
            json.dump(best, f, indent=2)
        print(f"[{name}] Final single-split summary: {best}\n")


# ----------------------------- CLI -------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cv", action="store_true", help="Run k-fold CV for each model")
    p.add_argument("--final", action="store_true", help="Run single 90/10 split training for each model")
    p.add_argument("--models", nargs="*", default=["simple_cnn", "resnet_cifar", "wrn28_10"],
                   help="Model names: simple_cnn | resnet_cifar | wrn28_10")
    # Allow quick overrides
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--base_lr", type=float, default=0.1)
    p.add_argument("--drop_out", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--save_dir", type=str, default="outputs", help="Directory to save model checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.cv or args.final, "Choose at least one of --cv or --final"

    cfg = CFG()
    # Apply quick overrides (without mutating the dataclass defaults globally)
    if args.epochs is not None:
        cfg = replace(cfg, epochs=args.epochs)
    if args.batch_size is not None:
        cfg = replace(cfg, batch_size=args.batch_size)
    if args.base_lr is not None:
        cfg = replace(cfg, base_lr=args.base_lr)
    if args.drop_out is not None:
        cfg = replace(cfg, dropout=args.drop_out)
    if args.weight_decay is not None:
        cfg = replace(cfg, weight_decay=args.weight_decay)
    if args.label_smoothing is not None:
        cfg = replace(cfg, label_smoothing=args.label_smoothing)
    if args.save_dir is not None:
        cfg = replace(cfg, save_dir=args.save_dir)

    print("Using config:\n", cfg)

    if args.cv:
        run_cross_validation(cfg, args.models)
    if args.final:
        run_final_single_split(cfg, args.models)


if __name__ == "__main__":
    main()
