import os, csv, json, time, itertools
from typing import Dict, List, Any
from dataclasses import replace

from src.utils import save_json
from src.config import CFG
from src.cv import run_cv

def grid_search(base_cfg: CFG, param_grid: Dict[str, List[Any]]):
    # build all param combinations
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    results = []
    t0 = time.time()
    print(f"Total combos: {len(combos)}")

    for i, values in enumerate(combos, start=1):
        cfg = replace(base_cfg)  # shallow copy dataclass
        # assign grid values
        for k, v in zip(keys, values):
            setattr(cfg, k, v)

        # unique dir per combo
        combo_name = "_".join([f"{k}={v}" for k, v in zip(keys, values)])
        cfg.save_dir = os.path.join(base_cfg.save_dir, f"gs_{combo_name}")
        os.makedirs(cfg.save_dir, exist_ok=True)
        save_json(vars(cfg), os.path.join(cfg.save_dir, "config.json"))

        print(f"\n=== [{i}/{len(combos)}] {combo_name} ===")
        metrics = run_cv(cfg)
        print(f"Val mean±std: {metrics['val_mean']:.4f} ± {metrics['val_std']:.4f}")

        rec = {"combo": combo_name, **{k: v for k, v in zip(keys, values)},
               **metrics, "time_min": (time.time() - t0)/60.0}
        results.append(rec)

        # append to global CSV
        csv_path = os.path.join(base_cfg.save_dir, "grid_results.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rec.keys()))
            if write_header: w.writeheader()
            w.writerow(rec)

    # sort by best val_mean
    results.sort(key=lambda r: r["val_mean"], reverse=True)
    print("\n===== GRID SUMMARY (top 5) =====")
    for r in results[:5]:
        print(f"{r['combo']}: mean={r['val_mean']:.4f} std={r['val_std']:.4f}")

    # save JSON summary
    save_json({"results": results}, os.path.join(base_cfg.save_dir, "grid_summary.json"))
    return results


# Grid search to optimize hyperparameters with based model is basic CNN
if __name__ == "__main__":
    # Initialize base config
    base_cfg = CFG()
    base_cfg.save_dir = "grid_search_results"
    base_cfg.model_name = "simple_cnn"

    param_grid = {
        "base_lr":       [0.005, 0.01, 0.05, 0.1],
        "weight_decay":  [3e-4, 1e-3],
        "dropout":       [0.3, 0.5],
        "label_smoothing":[0.1],
        "batch_size":    [256],   
        "epochs":        [120],    # try 100–150 once you find a good region
        "n_splits":      [3],     # use 5 for final measurement
    }

    results = grid_search(base_cfg, param_grid)
    best = results[0]
    print("\nBest combo:")
    print(best)