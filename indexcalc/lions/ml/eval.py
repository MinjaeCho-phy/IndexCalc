"""LIONS Task 1 evaluation — threshold-based metrics + sample inspection.

Reads a trained checkpoint and a JSON split, runs inference, and reports:
- per-group: accuracy / precision / recall / F1 at threshold τ
- confusion matrix per group
- "all-correct" rate (all heads simultaneously right)
- top-K misclassified samples with their IR expression
- a few correctly classified samples for sanity
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Sequence

from indexcalc.lions.ml import _require_torch
_require_torch()

import torch
from torch.utils.data import DataLoader

from indexcalc.lions import load_dataset
from indexcalc.lions.ml.features import GROUP_ORDER, num_relations
from indexcalc.lions.ml.models import RGCNClassifier
from indexcalc.lions.ml.datasets import LionsPyGDataset, collate_pyg


# ─── Metric primitives ───────────────────────────────────


def _confusion(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """Binary confusion matrix entries: TP, FP, TN, FN."""
    y_true = y_true.bool()
    y_pred = y_pred.bool()
    tp = (y_true & y_pred).sum().item()
    tn = (~y_true & ~y_pred).sum().item()
    fp = (~y_true & y_pred).sum().item()
    fn = (y_true & ~y_pred).sum().item()
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def _from_confusion(c: dict) -> dict:
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    total = tp + fp + tn + fn
    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        **c, "total": total,
    }


# ─── Main inference loop ─────────────────────────────────


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        ys.append(batch.y.cpu())
        ps.append(torch.sigmoid(logits).cpu())
    return torch.cat(ys, 0), torch.cat(ps, 0)


def evaluate_split(
    model_path: Path, data_path: Path, *,
    threshold: float = 0.5, device: str = "cpu",
    hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1,
):
    ds = LionsPyGDataset(data_path)
    print(f"Loaded {len(ds)} samples from {data_path}", flush=True)

    model = RGCNClassifier(
        hidden_dim=hidden_dim,
        num_relations=num_relations(),
        num_layers=num_layers,
        dropout=dropout,
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        collate_fn=collate_pyg)
    y_true, y_score = collect_predictions(model, loader, device)
    y_pred = (y_score >= threshold).float()

    # Per-group metrics
    per_group = {}
    for i, g in enumerate(GROUP_ORDER):
        c = _confusion(y_true[:, i], y_pred[:, i])
        per_group[g] = _from_confusion(c)

    # Joint ("all-3-heads-correct") rate
    all_correct = (y_pred == y_true).all(dim=1).float().mean().item()

    # Top-1 misclassified per group (by score margin = |p - threshold|)
    return {
        "threshold": threshold,
        "n_samples": len(ds),
        "per_group": per_group,
        "all_correct_rate": all_correct,
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_pred,
        "samples": load_dataset(data_path),
    }


# ─── Printing helpers ────────────────────────────────────


def print_report(res: dict):
    print(f"\n══ Test recognition (n={res['n_samples']}, τ={res['threshold']}) ══")
    print(f"{'Group':<10} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}  "
          f"{'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    for g in GROUP_ORDER:
        m = res["per_group"][g]
        print(f"{g:<10} {m['accuracy']:7.4f} {m['precision']:7.4f} "
              f"{m['recall']:7.4f} {m['f1']:7.4f}  "
              f"{m['TP']:4d} {m['FP']:4d} {m['TN']:4d} {m['FN']:4d}")
    print(f"\nALL-3-HEADS-CORRECT rate: {res['all_correct_rate']:.4f} "
          f"({int(res['all_correct_rate'] * res['n_samples'])}/"
          f"{res['n_samples']} samples)")


def print_provenance_breakdown(res: dict):
    """Per-provenance accuracy + all-3-correct rate.

    Reveals whether hard negatives (e.g. ``hard_negative_n3``) are
    actually harder than wrong-rep ones (``negative``).
    """
    y_true = res["y_true"]
    y_pred = res["y_pred"]
    samples = res["samples"]
    from collections import defaultdict
    by_prov: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_prov[s.provenance].append(i)
    print("\n── Per-provenance accuracy ──")
    print(f"{'provenance':<22} {'n':>5}  "
          f"{'SU(2)':>8} {'U(1)_Y':>8} {'Lorentz':>8}  "
          f"{'all3':>8}")
    for prov in sorted(by_prov):
        idx = by_prov[prov]
        if not idx:
            continue
        n = len(idx)
        acc_per_group = []
        for j in range(len(GROUP_ORDER)):
            correct = (y_pred[idx, j] == y_true[idx, j]).float().mean()
            acc_per_group.append(correct.item())
        all3 = (y_pred[idx] == y_true[idx]).all(dim=1).float().mean().item()
        cells = "  ".join(f"{a:8.4f}" for a in acc_per_group)
        print(f"{prov:<22} {n:>5d}  {cells}  {all3:>8.4f}")


def print_examples(res: dict, n: int = 5):
    """Show n correctly classified + n misclassified samples (if any)."""
    y_true = res["y_true"]
    y_pred = res["y_pred"]
    y_score = res["y_score"]
    samples = res["samples"]

    correct = (y_pred == y_true).all(dim=1)
    wrong = ~correct
    print(f"\n── Correct examples (top {n}) ──────────")
    correct_idx = correct.nonzero(as_tuple=True)[0].tolist()[:n]
    for i in correct_idx:
        s = samples[i]
        truth = {g: bool(y_true[i, j].item())
                 for j, g in enumerate(GROUP_ORDER)}
        scores = {g: round(y_score[i, j].item(), 3)
                  for j, g in enumerate(GROUP_ORDER)}
        print(f"  #{i:4d}  provenance={s.provenance:11s}  "
              f"truth={truth}")
        print(f"         scores={scores}")
        print(f"         IR (first 120 chars): {repr(s.expr)[:120]}")

    if wrong.any():
        print(f"\n── Misclassified examples (top {n}) ──")
        wrong_idx = wrong.nonzero(as_tuple=True)[0].tolist()[:n]
        for i in wrong_idx:
            s = samples[i]
            truth = {g: bool(y_true[i, j].item())
                     for j, g in enumerate(GROUP_ORDER)}
            pred = {g: bool(y_pred[i, j].item())
                    for j, g in enumerate(GROUP_ORDER)}
            scores = {g: round(y_score[i, j].item(), 3)
                      for j, g in enumerate(GROUP_ORDER)}
            print(f"  #{i:4d}  provenance={s.provenance:11s}")
            print(f"         truth={truth}")
            print(f"         pred ={pred}")
            print(f"         scores={scores}")
            print(f"         IR (first 120 chars): {repr(s.expr)[:120]}")
    else:
        print(f"\n── No misclassified samples at τ={res['threshold']} ──")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path,
                   default=Path.home() / "Minjae/LIONS/data/v1/runs/best_model.pt")
    p.add_argument("--data", type=Path,
                   default=Path.home() / "Minjae/LIONS/data/v1/test.json")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--examples", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    args = p.parse_args()

    res = evaluate_split(
        args.model, args.data,
        threshold=args.threshold,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
    )
    print_report(res)
    print_provenance_breakdown(res)
    print_examples(res, n=args.examples)


if __name__ == "__main__":
    main()
