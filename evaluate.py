import torch
import numpy as np
import json, os
from typing import Dict, List, Any, Optional

from src.model import build_model
from src.data import get_test_loader
from src.config import CFG

CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
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
    """
    Evaluate a classifier and report overall metrics + per-class breakdown.

    Returns a dict with:
        - loss, accuracy
        - confusion_matrix (numpy array [num_classes, num_classes])
        - per_class: list of dicts (label, support, tp, fp, fn, precision, recall, f1, acc)
        - macro: macro-averaged precision/recall/f1
        - micro: micro-averaged precision/recall/f1 (same as overall acc for single-label tasks)
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    if label_names is None:
        label_names = [str(i) for i in range(num_classes)]

    # Confusion matrix: rows = actual (y), cols = predicted (y_hat)
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

            # Update confusion matrix
            for t, p in zip(y.view(-1).cpu(), preds.view(-1).cpu()):
                conf[t.long(), p.long()] += 1

    avg_loss = running_loss / max(1, total)
    accuracy = correct / max(1, total)

    # Per-class stats
    conf_np = conf.numpy()
    tp = np.diag(conf_np)
    support = conf_np.sum(axis=1)            # actual count per class
    pred_sum = conf_np.sum(axis=0)           # predicted count per class
    fp = pred_sum - tp
    fn = support - tp

    # Avoid divide-by-zero
    def safe_div(n, d):
        return float(n) / float(d) if d > 0 else 0.0

    per_class: List[Dict[str, Any]] = []
    precisions, recalls, f1s = [], [], []

    for i in range(num_classes):
        prec = safe_div(tp[i], pred_sum[i])
        rec  = safe_div(tp[i], support[i])
        f1   = safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) > 0 else 0.0
        acc_i = safe_div(tp[i], support[i])  # same as recall for single-label
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
            "acc": acc_i,
        })
        precisions.append(prec); recalls.append(rec); f1s.append(f1)

    macro = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall":    float(np.mean(recalls))    if recalls    else 0.0,
        "f1":        float(np.mean(f1s))        if f1s        else 0.0,
    }
    # Micro for single-label multi-class = overall accuracy for precision/recall/f1
    micro = {"precision": accuracy, "recall": accuracy, "f1": accuracy}

    results = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "confusion_matrix": conf_np,  # shape [C, C], rows=actual, cols=pred
        "per_class": per_class,
        "macro": macro,
        "micro": micro,
        "total": total,
    }

    if print_table:
        # Pretty print per-class table
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


def load_model(model_name: str):
    path = f"trained_model/{model_name}_best.pt"
    checkpoint = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=10).to("cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def main():
    model_names = ["simple_cnn", "resnet_cifar"]
    cfg = CFG()  
    for model_name in model_names:
        model = load_model(model_name)
        test_loader = get_test_loader(cfg, data_root=cfg.data_root)
        criterion = torch.nn.CrossEntropyLoss()
        res = evaluate(
            model, dataloader=test_loader, criterion=criterion, device=cfg.device,
            num_classes=10, label_names=CIFAR10_LABELS, print_table=True
        )
        # Save to JSON file
        # os.makedirs("results", exist_ok=False)
        # with open(f"results/{model_name}_results.json", "w") as f:
        #     json.dump(res, f, indent=4)
        # print(f"Results saved for {model_name} in results/{model_name}_results.json")


if __name__ == "__main__":
    main()