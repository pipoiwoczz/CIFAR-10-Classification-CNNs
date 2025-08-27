import torch
import random
import numpy as np
import os, json, math
from typing import Dict, Any

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.wait = 0
        self.stop = False
    def step(self, metric: float) -> bool:
        improved = metric > self.best + self.min_delta
        if improved:
            self.best = metric; self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience: self.stop = True
        return improved

def cosine_with_warmup_lambda(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


