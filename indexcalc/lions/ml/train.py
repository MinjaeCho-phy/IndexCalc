"""LIONS Task 1 training loop — R-GCN baseline.

Reads JSON splits produced by ``LIONS/scripts/build_v1_toy.py``, builds
the PyG dataset, and trains ``RGCNClassifier`` with BCEWithLogitsLoss +
per-group ``pos_weight`` (computed from the training split label
histogram).

Per-epoch metrics: macro AUC across groups, per-group AUC, train/val loss.

Designed to run on CPU for v1-toy (<2 min/epoch at hidden=64).
"""

from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

from indexcalc.lions.ml import _require_torch
_require_torch()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from indexcalc.lions.ml.datasets import LionsPyGDataset, collate_pyg
from indexcalc.lions.ml.features import GROUP_ORDER, num_relations
from indexcalc.lions.ml.models import RGCNClassifier


# ─── Metrics ─────────────────────────────────────────────


def auc_roc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """Single-array binary AUC (no sklearn dep)."""
    y_true = y_true.flatten().cpu().numpy()
    y_score = y_score.flatten().cpu().numpy()
    # Need at least one positive and one negative.
    pos_n = int((y_true == 1).sum())
    neg_n = int((y_true == 0).sum())
    if pos_n == 0 or neg_n == 0:
        return float("nan")
    # Rank-based Mann–Whitney AUC (ranks ascending: lowest score → 1).
    import numpy as np
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    pos_ranks = ranks[y_true == 1]
    auc = (pos_ranks.sum() - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n)
    return float(auc)


def per_group_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> dict:
    """y_true / y_score: [N, G]. Returns {group: auc}."""
    out = {}
    for i, g in enumerate(GROUP_ORDER):
        out[g] = auc_roc(y_true[:, i], y_score[:, i])
    return out


# ─── Pos-weight from label histogram ─────────────────────


def compute_pos_weight(dataset: LionsPyGDataset) -> torch.Tensor:
    """For each group, pos_weight = #negatives / #positives in training."""
    G = len(GROUP_ORDER)
    pos = torch.zeros(G)
    neg = torch.zeros(G)
    for d in dataset._data:
        for i in range(G):
            if d.y_mask[0, i].item() < 0.5:
                continue
            if d.y[0, i].item() > 0.5:
                pos[i] += 1
            else:
                neg[i] += 1
    # Avoid div-by-zero.
    pw = torch.where(pos > 0, neg / pos.clamp(min=1.0),
                     torch.ones_like(pos))
    return pw


# ─── Train / eval loops ──────────────────────────────────


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, batch.y, batch.y_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    all_y, all_p = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = loss_fn(logits, batch.y, batch.y_mask)
        total += loss.item() * batch.num_graphs
        n += batch.num_graphs
        all_y.append(batch.y.cpu())
        all_p.append(torch.sigmoid(logits).cpu())
    y = torch.cat(all_y, dim=0)
    p = torch.cat(all_p, dim=0)
    return total / max(n, 1), per_group_auc(y, p)


# ─── Masked BCE wrapper ──────────────────────────────────


class MaskedBCEWithPosWeight(nn.Module):
    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, y, mask):
        # logits / y / mask: [B, G]
        per_sample = nn.functional.binary_cross_entropy_with_logits(
            logits, y,
            pos_weight=self.pos_weight.to(logits.device),
            reduction="none",
        )
        masked = per_sample * mask
        denom = mask.sum().clamp(min=1.0)
        return masked.sum() / denom


# ─── Main ────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path,
                   default=Path.home() / "Minjae/LIONS/data/v1-toy")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=Path.home() / "Minjae/LIONS/data/v1-toy/runs")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading datasets from {args.data_dir} ...", flush=True)
    train_ds = LionsPyGDataset(args.data_dir / "train.json")
    val_ds   = LionsPyGDataset(args.data_dir / "val.json")
    test_ds  = LionsPyGDataset(args.data_dir / "test.json")
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
          flush=True)

    pos_w = compute_pos_weight(train_ds)
    print(f"  pos_weight per group: "
          f"{dict(zip(GROUP_ORDER, [round(v.item(), 3) for v in pos_w]))}",
          flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_pyg)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_pyg)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_pyg)

    device = torch.device(args.device)
    model = RGCNClassifier(
        hidden_dim=args.hidden_dim,
        num_relations=num_relations(),
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = MaskedBCEWithPosWeight(pos_w).to(device)

    args.out.mkdir(parents=True, exist_ok=True)
    log = []
    best_val_auc = -math.inf
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
        )
        val_loss, val_auc = evaluate(model, val_loader, loss_fn, device)
        macro_auc = sum(v for v in val_auc.values()
                        if not math.isnan(v)) / max(
            sum(1 for v in val_auc.values() if not math.isnan(v)), 1)
        dt = time.time() - t0
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auc": macro_auc,
            **{f"val_auc_{g}": val_auc[g] for g in GROUP_ORDER},
            "time_s": round(dt, 2),
        }
        log.append(log_entry)
        auc_str = " ".join(
            f"{g}={val_auc[g]:.3f}" for g in GROUP_ORDER
        )
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  macro_AUC={macro_auc:.3f}  "
              f"{auc_str}  ({dt:.1f}s)", flush=True)
        if macro_auc > best_val_auc:
            best_val_auc = macro_auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    # Test with best checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_auc = evaluate(model, test_loader, loss_fn, device)
    print(f"\nTEST  loss={test_loss:.4f}  "
          f"per-group AUC: "
          f"{dict((g, round(test_auc[g], 4)) for g in GROUP_ORDER)}",
          flush=True)

    (args.out / "log.json").write_text(json.dumps({
        "config": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "dropout": args.dropout, "seed": args.seed,
        },
        "n_params": n_params,
        "pos_weight": {g: float(pos_w[i]) for i, g in enumerate(GROUP_ORDER)},
        "epochs": log,
        "best_val_macro_auc": best_val_auc,
        "test": {"loss": test_loss,
                 "auc": {g: test_auc[g] for g in GROUP_ORDER}},
    }, indent=2))
    if best_state is not None:
        torch.save(best_state, args.out / "best_model.pt")
    print(f"\nLog saved → {args.out}/log.json", flush=True)


if __name__ == "__main__":
    main()
