"""Dataset assembly — turn enumerated candidates into labeled samples.

A labeler runs ``apply_generator + simplify`` for each (group, generator)
and records whether the result resolves to ``ZeroTensor`` (= invariant).

Multi-group labels make each sample a row like:
    {"SU(2)": True, "U(1)_Y": True, "Lorentz": False}
"""

from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Optional

from indexcalc.core.tensor import TensorExpr
from indexcalc.core.generator import Generator
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify
from indexcalc.core.variation import ZeroTensor

from indexcalc.lions.enumerate import EnumeratedSample


@dataclass
class LabeledSample:
    expr: TensorExpr
    labels: dict[str, bool]
    mass_dim: float
    field_counts: dict[str, int]
    partial_count: int
    invariant_counts: dict[str, int] = dc_field(default_factory=dict)
    provenance: str = "enumerated"


def label_expression(
    expr: TensorExpr, generators: dict[str, Generator],
) -> dict[str, bool]:
    """Apply each generator, simplify, and label by ZeroTensor outcome."""
    out: dict[str, bool] = {}
    for group_name, gen in generators.items():
        delta = apply_generator(expr, gen)
        final = simplify(delta)
        out[group_name] = isinstance(final, ZeroTensor)
    return out


def label_samples(
    samples: list[EnumeratedSample],
    generators: dict[str, Generator],
) -> list[LabeledSample]:
    """Label every enumerated sample against every registered generator."""
    out: list[LabeledSample] = []
    for s in samples:
        labels = label_expression(s.expr, generators)
        out.append(LabeledSample(
            expr=s.expr,
            labels=labels,
            mass_dim=s.mass_dim,
            field_counts=dict(s.field_counts),
            partial_count=s.partial_count,
            invariant_counts=dict(getattr(s, "invariant_counts", {}) or {}),
        ))
    return out
