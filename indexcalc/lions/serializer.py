"""LIONS dataset persistence — JSON round-trip for ``LabeledSample`` lists.

D8a foundation for the rest of the D8 pipeline (negative / augmentation /
scale). Design: ``notes/d8a_persistence.md`` in the LIONS repo.

What round-trips:
- ``Tensor`` (all slot metadata: antisym/sym/traceless/transverse, reps,
  statistics) and the index-level ``Index`` + ``IndexSpace``.
- ``TensorProduct``, ``TensorSum``, ``ScalarMul`` (real / int / complex),
  ``PartialDeriv``, ``ZeroTensor``.
- ``LabeledSample`` wrapper.

Out of scope (v1):
- ``CovariantDeriv`` — needs ``Connection`` serialization, not used by
  LIONS dataset surface area yet. Raises ``NotImplementedError``.
- ``Generator`` / ``FieldRegistry`` — loaders supply these from presets;
  the dataset stores results, not the oracle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from indexcalc.core.index import Index, IndexSpace
from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.deriv import PartialDeriv, CovariantDeriv
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.scalar_function import ScalarFunction
from indexcalc.adm import TimeDeriv

from indexcalc.lions.dataset import LabeledSample


SCHEMA_VERSION = 1


# ─── IndexSpace ──────────────────────────────────────────


def space_to_dict(space: IndexSpace) -> dict:
    return {
        "dim": space.dim,
        "indices": space.indices,
        "metric": space.metric,
    }


def space_from_dict(name: str, d: dict) -> IndexSpace:
    return IndexSpace(
        name=name,
        dim=d["dim"],
        indices=d.get("indices", ""),
        metric=d.get("metric", ""),
    )


# ─── Index ────────────────────────────────────────────────


def index_to_dict(idx: Index) -> dict:
    return {
        "name": idx.name,
        "space": idx.space.name,
        "position": idx.position,
    }


def index_from_dict(d: dict, spaces: dict[str, IndexSpace]) -> Index:
    space_name = d["space"]
    if space_name not in spaces:
        raise KeyError(
            f"unknown IndexSpace {space_name!r} in dataset header "
            f"(known: {sorted(spaces)})"
        )
    return Index(d["name"], spaces[space_name], d["position"])


# ─── Scalar (int / float / complex) ─────────────────────


def _scalar_to_dict(s):
    if isinstance(s, complex):
        return {"re": s.real, "im": s.imag}
    return s  # int / float pass through JSON natively


def _scalar_from_dict(s):
    if isinstance(s, dict) and "re" in s and "im" in s:
        return complex(s["re"], s["im"])
    return s


# ─── TensorExpr ──────────────────────────────────────────


def expr_to_dict(expr: TensorExpr) -> dict:
    if isinstance(expr, Tensor):
        return {
            "type": "Tensor",
            "name": expr.name,
            "indices": [index_to_dict(i) for i in expr.indices],
            "antisymmetric_pairs": [list(p) for p in expr.antisymmetric_pairs],
            "symmetric_pairs": [list(p) for p in expr.symmetric_pairs],
            "traceless": [list(p) for p in expr.traceless],
            "transverse": list(expr.transverse),
            "reps": dict(expr.reps),
            "statistics": expr.statistics,
        }
    if isinstance(expr, TensorProduct):
        return {
            "type": "TensorProduct",
            "left": expr_to_dict(expr.left),
            "right": expr_to_dict(expr.right),
        }
    if isinstance(expr, TensorSum):
        return {
            "type": "TensorSum",
            "left": expr_to_dict(expr.left),
            "right": expr_to_dict(expr.right),
        }
    if isinstance(expr, ScalarMul):
        return {
            "type": "ScalarMul",
            "scalar": _scalar_to_dict(expr.scalar),
            "expr": expr_to_dict(expr.expr),
        }
    if isinstance(expr, PartialDeriv):
        return {
            "type": "PartialDeriv",
            "expr": expr_to_dict(expr.expr),
            "deriv_index": index_to_dict(expr.deriv_index),
        }
    if isinstance(expr, TimeDeriv):
        return {
            "type": "TimeDeriv",
            "expr": expr_to_dict(expr.expr),
        }
    if isinstance(expr, ScalarFunction):
        return {
            "type": "ScalarFunction",
            "name": expr.name,
            "arg": expr_to_dict(expr.arg),
        }
    if isinstance(expr, ZeroTensor):
        return {
            "type": "ZeroTensor",
            "free_indices": [index_to_dict(i) for i in expr.free_indices],
        }
    if isinstance(expr, CovariantDeriv):
        raise NotImplementedError(
            "CovariantDeriv serialization is deferred (D8a v1 scope). "
            "Connection metadata round-trip not implemented."
        )
    raise TypeError(f"unsupported expr type {type(expr).__name__}")


def expr_from_dict(d: dict, spaces: dict[str, IndexSpace]) -> TensorExpr:
    t = d["type"]
    if t == "Tensor":
        return Tensor(
            d["name"],
            [index_from_dict(i, spaces) for i in d["indices"]],
            antisymmetric_pairs=[tuple(p) for p in d.get("antisymmetric_pairs", [])],
            symmetric_pairs=[tuple(p) for p in d.get("symmetric_pairs", [])],
            traceless=[tuple(p) for p in d.get("traceless", [])],
            transverse=list(d.get("transverse", [])),
            reps=dict(d.get("reps", {})),
            statistics=d.get("statistics", "bosonic"),
        )
    if t == "TensorProduct":
        return TensorProduct(
            expr_from_dict(d["left"], spaces),
            expr_from_dict(d["right"], spaces),
        )
    if t == "TensorSum":
        return TensorSum(
            expr_from_dict(d["left"], spaces),
            expr_from_dict(d["right"], spaces),
        )
    if t == "ScalarMul":
        return ScalarMul(
            _scalar_from_dict(d["scalar"]),
            expr_from_dict(d["expr"], spaces),
        )
    if t == "PartialDeriv":
        return PartialDeriv(
            expr_from_dict(d["expr"], spaces),
            index_from_dict(d["deriv_index"], spaces),
        )
    if t == "TimeDeriv":
        return TimeDeriv(expr_from_dict(d["expr"], spaces))
    if t == "ScalarFunction":
        return ScalarFunction(d["name"], expr_from_dict(d["arg"], spaces))
    if t == "ZeroTensor":
        return ZeroTensor(
            [index_from_dict(i, spaces) for i in d.get("free_indices", [])],
        )
    raise ValueError(f"unknown expr type {t!r}")


# ─── Spaces collector ────────────────────────────────────


def collect_spaces(expr: TensorExpr) -> dict[str, IndexSpace]:
    """Walk expr and collect every IndexSpace referenced by an Index."""
    out: dict[str, IndexSpace] = {}

    def visit_index(idx: Index):
        out.setdefault(idx.space.name, idx.space)

    def walk(e: TensorExpr):
        if isinstance(e, Tensor):
            for i in e.indices:
                visit_index(i)
        elif isinstance(e, (TensorProduct, TensorSum)):
            walk(e.left); walk(e.right)
        elif isinstance(e, ScalarMul):
            walk(e.expr)
        elif isinstance(e, PartialDeriv):
            walk(e.expr); visit_index(e.deriv_index)
        elif isinstance(e, TimeDeriv):
            walk(e.expr)
        elif isinstance(e, ScalarFunction):
            walk(e.arg)
        elif isinstance(e, ZeroTensor):
            for i in e.free_indices:
                visit_index(i)
        elif isinstance(e, CovariantDeriv):
            raise NotImplementedError(
                "CovariantDeriv space collection deferred (D8a v1)."
            )
    walk(expr)
    return out


# ─── LabeledSample ───────────────────────────────────────


def sample_to_dict(s: LabeledSample) -> dict:
    return {
        "expr": expr_to_dict(s.expr),
        "labels": dict(s.labels),
        "mass_dim": s.mass_dim,
        "field_counts": dict(s.field_counts),
        "partial_count": s.partial_count,
        "invariant_counts": dict(s.invariant_counts),
        "provenance": s.provenance,
    }


def sample_from_dict(d: dict, spaces: dict[str, IndexSpace]) -> LabeledSample:
    return LabeledSample(
        expr=expr_from_dict(d["expr"], spaces),
        labels=dict(d["labels"]),
        mass_dim=d["mass_dim"],
        field_counts=dict(d.get("field_counts", {})),
        partial_count=d.get("partial_count", 0),
        invariant_counts=dict(d.get("invariant_counts", {})),
        provenance=d.get("provenance", "enumerated"),
    )


# ─── Top-level save / load ───────────────────────────────


def save_dataset(samples: list[LabeledSample], path: str | Path) -> None:
    """Write samples + their referenced IndexSpaces to a JSON file."""
    all_spaces: dict[str, IndexSpace] = {}
    sample_dicts = []
    for s in samples:
        for nm, sp in collect_spaces(s.expr).items():
            if nm in all_spaces and all_spaces[nm].dim != sp.dim:
                raise ValueError(
                    f"IndexSpace {nm!r} appears with conflicting dim "
                    f"({all_spaces[nm].dim} vs {sp.dim}) across samples"
                )
            all_spaces.setdefault(nm, sp)
        sample_dicts.append(sample_to_dict(s))

    payload = {
        "version": SCHEMA_VERSION,
        "spaces": {nm: space_to_dict(sp) for nm, sp in all_spaces.items()},
        "samples": sample_dicts,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def load_dataset(path: str | Path) -> list[LabeledSample]:
    """Read a JSON dataset file written by ``save_dataset``."""
    raw = json.loads(Path(path).read_text())
    version = raw.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"dataset schema version mismatch: expected {SCHEMA_VERSION}, "
            f"got {version!r}"
        )
    spaces = {
        nm: space_from_dict(nm, d) for nm, d in raw["spaces"].items()
    }
    return [sample_from_dict(s, spaces) for s in raw["samples"]]
