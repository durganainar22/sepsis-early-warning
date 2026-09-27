"""Step 5b - GRU on the raw hourly sequence (hospital A only).

A one-directional GRU reads a patient's stay hour by hour and emits a sepsis probability
at EVERY hour, from its memory of hours <= t. It is causal by construction - it has no
path to later hours - and that is still tested, not assumed (truncation test at the end).

WHERE THIS DIVERGES FROM THE TABULAR MODELS:
  - no hand-built history. XGBoost got 164 features summarizing the past; the GRU gets the
    raw per-hour channels (value, mask, time-since) and must learn its own summaries.
  - one model call per PATIENT, not per row. Hours within a stay share the recurrent state,
    so the network sees the order of events, which a per-row model never does.
  - inputs are standardized (build_sequences.py); a tree never needed it.

Protocol, identical to XGBoost (PROJECT_PLAN.md): 40 configurations x 3 seeds selected on
mean val PR-AUC, winner re-run over 5 seeds; early stopping on val PR-AUC over scored
hours; plain BCE with NO pos_weight; loss and metrics over scored hours only.

Usage:  python src/train_gru.py [--n-iter 40] [--time-one]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "data" / "processed" / "seq_a.npz"
MODELS, REPORTS = ROOT / "models", ROOT / "reports"

SEEDS = [0, 1, 2, 3, 4]
SEARCH_SEEDS = [0, 1, 2]
SEARCH_SEED = 20260926
MAX_EPOCHS, PATIENCE = 40, 4
GRAD_CLIP = 1.0  # recurrent nets can take one huge gradient step on a long stay; clip it


def sample_config(rng: np.random.Generator) -> dict:
    return {
        "hidden": int(rng.choice([32, 64, 128])),
        "layers": int(rng.choice([1, 2])),
        "dropout": float(rng.uniform(0.0, 0.5)),
        "lr": float(np.exp(rng.uniform(np.log(3e-4), np.log(3e-3)))),
        "weight_decay": float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-3)))),
        "batch_size": int(rng.choice([32, 64, 128])),
    }


class GRUModel(nn.Module):
    def __init__(self, n_in: int, cfg: dict):
        super().__init__()
        self.gru = nn.GRU(n_in, cfg["hidden"], num_layers=cfg["layers"], batch_first=True,
                          dropout=cfg["dropout"] if cfg["layers"] > 1 else 0.0)
        self.drop = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(cfg["hidden"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)          # (batch, time, hidden); right-padding never reaches back
        return self.head(self.drop(h)).squeeze(-1)


class Data:
    """Ragged patient sequences for one split, padded per batch."""

    def __init__(self, z, split: str):
        keep = np.flatnonzero(z["split"] == split)
        off = z["offsets"]
        self.slices = [(off[i], off[i + 1]) for i in keep]
        self.pid = z["pid"][keep]
        self.x, self.y, self.sc = z["x"], z["y"], z["scored"]
        self.len = np.array([b - a for a, b in self.slices])

    def batches(self, bs: int, rng: np.random.Generator | None):
        # Bucket by length so a batch pads to similar lengths; shuffle BATCH order for training.
        order = np.argsort(self.len, kind="stable")
        chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
        if rng is not None:
            rng.shuffle(chunks)
        for idx in chunks:
            T = self.len[idx].max()
            x = np.zeros((len(idx), T, self.x.shape[1]), dtype="float32")
            y = np.zeros((len(idx), T), dtype="float32"); w = np.zeros((len(idx), T), dtype="float32")
            for r, i in enumerate(idx):
                a, b = self.slices[i]
                x[r, :b - a], y[r, :b - a], w[r, :b - a] = self.x[a:b], self.y[a:b], self.sc[a:b]
            yield idx, torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)


@torch.no_grad()
def predict(model: nn.Module, d: Data) -> dict:
    """Scored-hour probabilities, in (patient, hour) order - the same order as features_a."""
    model.eval()
    probs = [None] * len(d.slices)
    for idx, x, _, w in d.batches(256, None):
        p = torch.sigmoid(model(x)).numpy()
        for r, i in enumerate(idx):
            probs[i] = p[r, :d.len[i]]
    p = np.concatenate(probs)
    a_rows = np.concatenate([np.arange(a, b) for a, b in d.slices])
    keep = d.sc[a_rows]
    hour = np.concatenate([np.arange(n) for n in d.len])
    return {"p": p[keep], "y": d.y[a_rows][keep].astype(int), "hour": hour[keep],
            "pid": np.repeat(d.pid, d.len)[keep]}


def train_one(cfg: dict, seed: int, tr: Data, va: Data) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GRUModel(tr.x.shape[1], cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    lossf = nn.BCEWithLogitsLoss(reduction="none")  # no pos_weight: same as the tabular models
    best, best_state, best_epoch, since = -1.0, None, 0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for _, x, y, w in tr.batches(cfg["batch_size"], rng):
            opt.zero_grad()
            loss = (lossf(model(x), y) * w).sum() / w.sum().clamp(min=1.0)  # scored hours only
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        pv = predict(model, va)
        score = ev.metrics(pv["y"], pv["p"])["pr_auc"]
        if score > best:
            best, best_epoch, since = score, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return {"model": model, "pr_auc": best, "epoch": best_epoch,
            "n_params": sum(p.numel() for p in model.parameters())}


def truncation_test(model: nn.Module, d: Data, n: int = 50) -> None:
    """Output at hour k must be identical whether or not hours > k exist."""
    model.eval()
    rng = np.random.default_rng(0)
    for i in rng.choice(len(d.slices), size=n, replace=False):
        a, b = d.slices[i]
        k = int(rng.integers(1, b - a + 1))
        with torch.no_grad():
            full = model(torch.from_numpy(d.x[a:b][None]))[0, :k]
            cut = model(torch.from_numpy(d.x[a:a + k][None]))[0]
        torch.testing.assert_close(full, cut, rtol=1e-5, atol=1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--time-one", action="store_true", help="time a single run and exit")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)

    z = dict(np.load(SEQ, allow_pickle=True))
    tr, va = Data(z, "train"), Data(z, "val")
    print(f"train {len(tr.slices):,} patients / {tr.len.sum():,} hours | val {len(va.slices):,} | "
          f"{tr.x.shape[1]} channels", flush=True)

    rng = np.random.default_rng(SEARCH_SEED)
    if args.time_one:
        cfg = sample_config(rng); t0 = time.perf_counter()
        r = train_one(cfg, 0, tr, va)
        truncation_test(r["model"], va)
        print(f"one run: {time.perf_counter() - t0:.0f}s, {r['epoch']} best epoch, "
              f"val PR-AUC {r['pr_auc']:.4f}, {r['n_params']:,} params, cfg {cfg}; truncation test passed")
        return

    trials, t_all = [], time.perf_counter()
    for i in range(args.n_iter):
        cfg = sample_config(rng); t0 = time.perf_counter()
        rs = [train_one(cfg, s, tr, va) for s in SEARCH_SEEDS]
        sc = [r["pr_auc"] for r in rs]
        trials.append({**cfg, "pr_auc_mean": float(np.mean(sc)), "pr_auc_sd": float(np.std(sc)),
                       "epochs": [r["epoch"] for r in rs], "n_params": rs[0]["n_params"],
                       "seconds": time.perf_counter() - t0})
        print(f"  [{i + 1:>2}/{args.n_iter}] PR-AUC {np.mean(sc):.4f} +-{np.std(sc):.4f}  "
              f"h={cfg['hidden']}x{cfg['layers']} do={cfg['dropout']:.2f} lr={cfg['lr']:.4f} "
              f"wd={cfg['weight_decay']:.1e} bs={cfg['batch_size']} -> ep {trials[-1]['epochs']}, "
              f"{rs[0]['n_params']:,}p, {time.perf_counter() - t0:.0f}s", flush=True)

    best = max(trials, key=lambda r: r["pr_auc_mean"])
    best_cfg = {k: best[k] for k in sample_config(np.random.default_rng(0))}
    print(f"\nSearch took {(time.perf_counter() - t_all) / 60:.1f} min. Best config: {best_cfg}")

    runs, final, epochs, val_preds = [], None, [], []
    for seed in SEEDS:
        r = train_one(best_cfg, seed, tr, va)
        pv = predict(r["model"], va)
        thr, _ = ev.best_utility_threshold(pv["pid"], pv["hour"], pv["y"], pv["p"])
        runs.append(ev.metrics(pv["y"], pv["p"], pv["pid"], pv["hour"], threshold=thr))
        epochs.append(r["epoch"]); val_preds.append(pv["p"])
        final = final or r
        print(f"  seed {seed}: PR-AUC {runs[-1]['pr_auc']:.4f}  utility {runs[-1]['utility']:.4f}  (epoch {r['epoch']})", flush=True)

    truncation_test(final["model"], va)
    print("  truncation test passed: outputs at hour k do not depend on hours > k")
    summary = ev.seed_summary(runs)
    ev.print_metrics("gru", summary)

    torch.save({"state_dict": final["model"].state_dict(), "config": best_cfg}, MODELS / "gru.pt")
    np.save(REPORTS / "step5_gru_val_preds.npy", np.array(val_preds, dtype="float32"))
    out = {"model": "gru", "best_config": best_cfg, "n_params": final["n_params"],
           "selected_on": "mean val pr_auc over 3 seeds",
           "search": {"n_iter": args.n_iter, "search_seed": SEARCH_SEED, "search_seeds": SEARCH_SEEDS,
                      "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
           "best_epoch_per_seed": epochs, "seeds": SEEDS, "val": summary, "trials": trials,
           "note": "Hospital A val only. Internal test and hospital B untouched until Step 6."}
    (REPORTS / "step5_gru.json").write_text(json.dumps(out, indent=2))
    print("\nWrote reports/step5_gru.json and models/gru.pt")


if __name__ == "__main__":
    main()
