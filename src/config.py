from dataclasses import dataclass
import torch

NUM_CLASSES = 10  # CIFAR-10 has 10 classes

@dataclass
class CFG:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    subsample_frac: float = 0.15   # ← use 15% of training set
    subsample_seed: int = 42       # for reproducibility

    # training
    epochs: int = 60                     # shorter for search; increase for final training
    batch_size: int = 128
    num_workers: int = 0                 # Windows-safe
    base_lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    dropout: float = 0.2
    use_amp: bool = True

    # early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # k-fold
    n_splits: int = 3                    # 3 for speed in search; use 5 later
    shuffle_folds: bool = True

    # schedule
    warmup_epochs: int = 5

    # directories
    data_root: str = "./data"
    save_root: str = "./run"
    save_dir: str = "./runs/cifar10_grid"

    # model 
    model_name: str = "simple_cnn"

    # # data augmentation
    # aug_policy: str = "basic"   # choices: "none","basic","randaugment","autoaugment","trivial"
    # randaugment_n: int = 2      # number of ops
    # randaugment_m: int = 9      # magnitude (0–10)
    # random_erasing_p: float = 0.0