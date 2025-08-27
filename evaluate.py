import os, json
import numpy as np
import torch
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt

from src.model import build_model
from src.config import CFG

# Try import; fall back to a local helper if not present
try:
    from src.data import get_test_loader
except ImportError:
    import torchvision as tv
    from torch.utils.data import DataLoader
    MEAN = [0.4914, 0.4822, 0.4465]
    STD  = [0.2023, 0.1994, 0.2010]
    def get_test_loader(cfg: CFG, data_root: str = "./data"):
        tf = tv.transforms.Compose([tv.transforms.ToTensor(),
                                    tv.transforms.Normalize(MEAN, STD)])
        test = tv.datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf)
        return DataLoader(test, batch_size=cfg.batch_size*2 if hasattr(cfg, "batch_size") else 256,
                          shuffle=False, num_workers=cfg.num_workers if hasattr(cfg, "num_workers") else 0)

CIFAR10_LABELS = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def evaluate(
    model,
    dataloader,
    criterion,
    device,
    num_classes: int = 10,
    label_names: Optional[List[str]] = None,
    print_table: bool = True,
) -> Dict[str, Any]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    if label_names is None:
        label_names = [str(i) for i in range(num_classes)]

    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * y.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            for t, p in zip(y.view(-1).cpu(), preds.view(-1).cpu()):
                conf[t.long(), p.long()] += 1

    avg_loss = running_loss / max(1, total)
    accuracy = correct / max(1, total)

    conf_np = conf.numpy()
    tp = np.diag(conf_np)
    support = conf_np.sum(axis=1)
    pred_sum = conf_np.sum(axis=0)
    fp = pred_sum - tp
    fn = support - tp

    def safe_div(n, d): return float(n) / float(d) if d > 0 else 0.0

    per_class: List[Dict[str, Any]] = []
    precisions, recalls, f1s = [], [], []
    for i in range(num_classes):
        prec = safe_div(tp[i], pred_sum[i])
        rec  = safe_div(tp[i], support[i])
        f1   = safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class.append({
            "label_idx": i,
            "label": label_names[i] if i < len(label_names) else str(i),
            "support": int(support[i]),
            "tp": int(tp[i]),
            "fp": int(fp[i]),
            "fn": int(fn[i]),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "acc": rec,  # same as recall for single-label tasks
        })
        precisions.append(prec); recalls.append(rec); f1s.append(f1)

    macro = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall":    float(np.mean(recalls))    if recalls    else 0.0,
        "f1":        float(np.mean(f1s))        if f1s        else 0.0,
    }
    micro = {"precision": accuracy, "recall": accuracy, "f1": accuracy}

    results = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "confusion_matrix": conf_np.tolist(),  # <-- JSON serializable
        "per_class": per_class,
        "macro": macro,
        "micro": micro,
        "total": total,
    }

    if print_table:
        header = f"{'idx':>3}  {'label':<12} {'sup':>5} {'tp':>5} {'fp':>5} {'fn':>5}  {'prec':>6} {'rec':>6} {'f1':>6}"
        print(header)
        print("-" * len(header))
        for r in per_class:
            print(f"{r['label_idx']:>3}  {r['label']:<12} {r['support']:>5} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5}  "
                  f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")
        print("-" * len(header))
        print(f"Overall Acc: {accuracy:.4f} | Macro P/R/F1: "
              f"{macro['precision']:.3f}/{macro['recall']:.3f}/{macro['f1']:.3f}")

    return results

# ---------- Saving helpers ----------

def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_per_class_csv(per_class: List[Dict[str, Any]], path: str):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["label_idx","label","support","tp","fp","fn","precision","recall","f1","acc"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in per_class:
            w.writerow(row)

def plot_confusion_matrix(conf: np.ndarray, labels: List[str], out_path: str, normalize: bool = True):
    """
    Save a confusion matrix image. If normalize=True, rows sum to 1.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cm = conf.astype(np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    plt.figure(figsize=(6.5, 5.5), dpi=150)
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=40, ha="right")
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def load_model(model_name: str):
    path = f"trained_model/{model_name}_best.pt"
    checkpoint = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    # Map aliases to your factory names if needed
    name_map = {"resnet18": "resnet_cifar"}  # resnet18 alias to your ResNetCIFAR builder
    build_name = name_map.get(model_name, model_name)
    model = build_model(build_name, num_classes=10)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def main():
    # Choose which models to evaluate (must match checkpoint names in trained_model/)
    model_names = ["resnet18", "simple_cnn", "resnet_cifar"]

    cfg = CFG()
    device = torch.device(cfg.device)
    test_loader = get_test_loader(cfg, data_root=cfg.data_root)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs("results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    for model_name in model_names:
        print(f"\n=== Evaluating {model_name} ===")
        model = load_model(model_name).to(device)

        res = evaluate(
            model, dataloader=test_loader, criterion=criterion, device=device,
            num_classes=10, label_names=CIFAR10_LABELS, print_table=True
        )

        # 1) Save JSON (confusion matrix stored as nested lists)
        json_path = f"results/{model_name}_results.json"
        save_json(res, json_path)
        print(f"Saved JSON: {json_path}")

        # 2) Save per-class CSV
        csv_path = f"reports/{model_name}_per_class.csv"
        save_per_class_csv(res["per_class"], csv_path)
        print(f"Saved per-class CSV: {csv_path}")

        # 3) Save confusion matrix image (normalized)
        cm_path = f"reports/{model_name}_cm.png"
        conf_np = np.array(res["confusion_matrix"])
        plot_confusion_matrix(conf_np, CIFAR10_LABELS, cm_path, normalize=True)
        print(f"Saved confusion matrix image: {cm_path}")

if __name__ == "__main__":
    main()
