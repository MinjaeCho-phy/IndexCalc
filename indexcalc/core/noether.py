"""Noether-style symmetry verification (Direction C, C0).

The catalog oracle (``probe.py``) only checks *internal* symmetries: a linear
generator with ``apply_generator(L) → 0`` exactly. But Noether symmetries —
including the velocity-dependent transformations behind *hidden* (dynamical)
symmetries — leave the Lagrangian invariant only up to a **total time
derivative**: ``δL = d/dt F`` for some boundary term F. This module adds that
recognizer, so a transformation can be verified as a symmetry even when δL ≠ 0.

Pieces:
- ``dt_expand``        — push ∂_t through products/sums (Leibniz); ∂_t of a
                         non-dynamical leaf (metric / invariant tensor) is 0.
- ``is_total_time_derivative`` — does ``δL = d/dt F`` for F = 0 or a supplied
                         candidate boundary term?
- ``verify_symmetry``  — apply a (possibly nonlinear/velocity-dependent)
                         generator to L, then test the total-derivative
                         condition. The candidate-F search is exactly where a
                         learned proposal prior (Direction C) would plug in.

Scope (C0): off-shell symmetries — δL = d/dt F as an algebraic identity. On-
shell (EOM-dependent) symmetries are a further lift (see notes).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional

from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.scalar_function import ScalarFunction
from indexcalc.core.generator import Generator
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify
from indexcalc.core.variation import ZeroTensor
from indexcalc.adm import TimeDeriv


def _is_constant_leaf(t: Tensor, field_names: frozenset[str]) -> bool:
    """A leaf is constant under ∂_t unless it is a named dynamical field.

    Invariant tensors (δ/η/ε/Ω/…) and couplings carry no field name, so they
    are background: ∂_t(δ_ij) = 0.
    """
    return t.name not in field_names


def _push_dt(inner: TensorExpr, field_names: frozenset[str]) -> TensorExpr:
    """Apply one ∂_t to an (already dt-expanded) expression, Leibniz-distributed."""
    if isinstance(inner, ScalarMul):
        return ScalarMul(inner.scalar, _push_dt(inner.expr, field_names))
    if isinstance(inner, TensorSum):
        return TensorSum(_push_dt(inner.left, field_names),
                         _push_dt(inner.right, field_names))
    if isinstance(inner, TensorProduct):
        return TensorSum(
            _push_dt(inner.left, field_names) * inner.right,
            inner.left * _push_dt(inner.right, field_names),
        )
    if isinstance(inner, ScalarFunction):
        # Chain rule: ∂_t f(I) = f'(I) ∂_t I. Mirrors apply_generator's
        # δf(I) = f'(I) δI so the variation and the boundary-term ∂_t agree
        # on the same f_prime symbol. ∂_t I = 0 ⇒ ∂_t f = 0.
        d_arg = _push_dt(inner.arg, field_names)
        if isinstance(d_arg, ZeroTensor):
            return ZeroTensor([])
        return TensorProduct(ScalarFunction(f"{inner.name}_prime", inner.arg), d_arg)
    if isinstance(inner, Tensor) and _is_constant_leaf(inner, field_names):
        return ZeroTensor(inner.free_indices)
    # dynamical field leaf, or an existing TimeDeriv(...) → one more ∂_t.
    return TimeDeriv(inner)


def dt_expand(expr: TensorExpr, field_names: Iterable[str]) -> TensorExpr:
    """Rewrite ``expr`` pushing every ∂_t inward via the Leibniz rule.

    ``field_names`` are the dynamical fields; ∂_t of any other leaf (metric,
    invariant tensor, coupling) is zero. Leaves the result in the same
    fully-distributed form ``apply_generator`` produces, so the two can be
    compared by ``simplify``.
    """
    fields = frozenset(field_names)
    if isinstance(expr, TimeDeriv):
        return _push_dt(dt_expand(expr.expr, fields), fields)
    if isinstance(expr, TensorSum):
        return TensorSum(dt_expand(expr.left, fields), dt_expand(expr.right, fields))
    if isinstance(expr, TensorProduct):
        return dt_expand(expr.left, fields) * dt_expand(expr.right, fields)
    if isinstance(expr, ScalarMul):
        return ScalarMul(expr.scalar, dt_expand(expr.expr, fields))
    if isinstance(expr, ScalarFunction):
        # No ∂_t to push here, but the argument may carry TimeDerivs to expand.
        return ScalarFunction(expr.name, dt_expand(expr.arg, fields))
    return expr


def _is_zero(expr: TensorExpr) -> bool:
    return isinstance(simplify(expr), ZeroTensor)


def is_total_time_derivative(
    delta_L: TensorExpr,
    field_names: Iterable[str],
    candidates: Optional[Iterable[TensorExpr]] = None,
) -> tuple[bool, Optional[TensorExpr]]:
    """Is ``delta_L = d/dt F`` for F = 0 or one of ``candidates``?

    Returns ``(True, F)`` on the first match (F is ``None`` for the exact-zero
    case), else ``(False, None)``. The candidate boundary terms are supplied by
    the caller — searching them is the learnable step in Direction C.
    """
    fields = frozenset(field_names)
    if _is_zero(delta_L):                       # δL = 0 exactly (internal sym)
        return True, None
    for F in (candidates or []):
        residual = TensorSum(delta_L, ScalarMul(-1.0, dt_expand(TimeDeriv(F), fields)))
        if _is_zero(residual):
            return True, F
    return False, None


@dataclass
class SymmetryResult:
    is_symmetry: bool
    exact: bool                       # True ⇔ δL = 0 (no boundary term needed)
    boundary_term: Optional[TensorExpr]
    delta_L: TensorExpr


def verify_symmetry(
    lagrangian: TensorExpr,
    generator: Generator,
    field_names: Iterable[str],
    boundary_candidates: Optional[Iterable[TensorExpr]] = None,
) -> SymmetryResult:
    """Verify a transformation is a symmetry of ``lagrangian``.

    Applies ``generator`` (a linear *or* velocity-dependent action — anything
    whose ``apply_to`` returns the variation δφ) via Leibniz, then checks
    ``δL = d/dt F`` for F = 0 or a supplied boundary candidate.
    """
    delta_L = apply_generator(lagrangian, generator)
    ok, F = is_total_time_derivative(delta_L, field_names, boundary_candidates)
    return SymmetryResult(is_symmetry=ok, exact=(ok and F is None),
                          boundary_term=F, delta_L=delta_L)
