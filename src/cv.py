import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
import os
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .data import make_fold_loaders
from .model import build_model
from .utils import cosine_with_warmup_lambda, EarlyStopping, set_seed
from .config import CFG, NUM_CLASSES

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

def train_one_fold(fold_id: int, cfg: CFG, train_loader, val_loader, outdir: str) -> float:
    model = build_model(cfg.model_name, num_classes=NUM_CLASSES, dropout=cfg.dropout).to(cfg.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.base_lr,
                                momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    lr_lambda = lambda e: cosine_with_warmup_lambda(e, cfg.epochs, cfg.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.device == "cuda" and cfg.use_amp))


    early = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(cfg.device == "cuda" and cfg.use_amp)):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, cfg.device)
        print(f"[Fold {fold_id}] Epoch {epoch+1:03d}/{cfg.epochs} | val_acc={val_acc:.4f} | val_loss={val_loss:.4f}")

        if early.step(val_acc):
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(outdir, f"fold{fold_id}_best.pt"))

        if early.stop:
            print(f"[Fold {fold_id}] Early stopping at epoch {epoch+1}. Best val acc={best_acc:.4f}")
            break

    return best_acc

def stratified_subsample_indices(y_all: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """
    Returns indices for a stratified subset with given fraction.
    If frac >= 1.0, returns all indices.
    """
    n = len(y_all)
    if frac >= 1.0:
        return np.arange(n)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
    (sub_idx, _), = sss.split(np.zeros(n), y_all)  # dummy X, stratify by y
    return sub_idx

def run_cv(cfg: CFG) -> Dict[str, Any]:
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # load base train once
    base_train = tv.datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=None)
    y_all = np.array(base_train.targets)

    sub_idx = stratified_subsample_indices(y_all, cfg.subsample_frac, cfg.subsample_seed)
    y_sub = y_all[sub_idx]

    print(f"[Subsample] Using {len(sub_idx)} / {len(y_all)} images "
          f"({100*len(sub_idx)/len(y_all):.1f}%) for hyperparameter tuning.")

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle_folds, random_state=cfg.seed)

    fold_accs: List[float] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_sub)), y_sub), start=1):
        fold_dir = os.path.join(cfg.save_dir, f"cv_tmp_fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        train_loader, val_loader = make_fold_loaders(base_train, train_idx, val_idx, cfg)
        best_val_acc = train_one_fold(fold, cfg, train_loader, val_loader, fold_dir)
        fold_accs.append(best_val_acc)

    return {
        "val_mean": float(np.mean(fold_accs)),
        "val_std": float(np.std(fold_accs)),
        "fold_accs": [float(a) for a in fold_accs],
    }