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
    # ``apply_generator`` leaves the variation of a velocity in the form
    # ∂_t(δφ) (e.g. δẋ = ∂_t(δx) for a velocity-dependent δx). Expand those ∂_t
    # by the same Leibniz rule used on the boundary term, so both sides of the
    # comparison are in one fully-distributed form (identity-preserving rewrite).
    delta_L = dt_expand(delta_L, fields)
    if _is_zero(delta_L):                       # δL = 0 exactly (internal sym)
        return True, None
    for F in (candidates or []):
        residual = TensorSum(delta_L, ScalarMul(-1.0, dt_expand(TimeDeriv(F), fields)))
        if _is_zero(residual):
            return True, F
    return False, None


# ─── Field-theory (∂_μ) Noether: δL = ∂_μ J^μ (total divergence) ──────
# Mirror of the 1D ∂_t verifier above, lifted to full spacetime: a symmetry of
# a field-theory Lagrangian leaves L invariant up to a total *divergence*
# ∂_μ J^μ (not just a total time derivative). Same Leibniz/constant-leaf/chain
# rule logic, with PartialDeriv carrying the derivative index μ.

def _push_div(inner: TensorExpr, field_names: frozenset[str],
              mu: "Index") -> TensorExpr:
    """Apply one ∂_μ to an (already div-expanded) expression, Leibniz."""
    from indexcalc.core.deriv import PartialDeriv
    if isinstance(inner, ScalarMul):
        return ScalarMul(inner.scalar, _push_div(inner.expr, field_names, mu))
    if isinstance(inner, TensorSum):
        return TensorSum(_push_div(inner.left, field_names, mu),
                         _push_div(inner.right, field_names, mu))
    if isinstance(inner, TensorProduct):
        return TensorSum(
            _push_div(inner.left, field_names, mu) * inner.right,
            inner.left * _push_div(inner.right, field_names, mu),
        )
    if isinstance(inner, ScalarFunction):
        d_arg = _push_div(inner.arg, field_names, mu)
        if isinstance(d_arg, ZeroTensor):
            return ZeroTensor([mu])
        return TensorProduct(ScalarFunction(f"{inner.name}_prime", inner.arg), d_arg)
    if isinstance(inner, Tensor) and getattr(inner, "is_coordinate", False):
        # ∂_μ x^ν = δ^ν_μ — 비-field leaf의 ∂=0 규칙의 유일한 예외(좌표).
        # eliminate_kronecker 가 δ^ν_μ X^μ → X^ν / 자기수축 δ^μ_μ=dim 을 처리.
        nu = inner.indices[0]
        return Tensor("delta", [nu, mu], reps={})
    if isinstance(inner, Tensor) and _is_constant_leaf(inner, field_names):
        return ZeroTensor(PartialDeriv(inner, mu).free_indices)
    # dynamical field leaf, or an existing PartialDeriv(...) → one more ∂_μ.
    return PartialDeriv(inner, mu)


def div_expand(expr: TensorExpr, field_names: Iterable[str]) -> TensorExpr:
    """Rewrite ``expr`` pushing every ∂_μ inward via Leibniz (spacetime).

    Field-theory analogue of ``dt_expand``: ∂_μ of a non-field leaf (metric,
    invariant tensor, coupling, constant translation vector) is zero; the
    ScalarFunction chain rule carries f'(I)∂_μI. Leaves the result in the same
    fully-distributed form ``apply_generator`` produces so the two compare by
    ``simplify``.
    """
    from indexcalc.core.deriv import PartialDeriv
    fields = frozenset(field_names)
    if isinstance(expr, PartialDeriv):
        return _push_div(div_expand(expr.expr, fields), fields, expr.deriv_index)
    if isinstance(expr, TensorSum):
        return TensorSum(div_expand(expr.left, fields), div_expand(expr.right, fields))
    if isinstance(expr, TensorProduct):
        return div_expand(expr.left, fields) * div_expand(expr.right, fields)
    if isinstance(expr, ScalarMul):
        return ScalarMul(expr.scalar, div_expand(expr.expr, fields))
    if isinstance(expr, ScalarFunction):
        return ScalarFunction(expr.name, div_expand(expr.arg, fields))
    return expr


def is_total_divergence(
    delta_L: TensorExpr,
    field_names: Iterable[str],
    deriv_index: "Index",
    candidates: Optional[Iterable[TensorExpr]] = None,
) -> tuple[bool, Optional[TensorExpr]]:
    """Is ``delta_L = ∂_μ J^μ`` for J=0 or one of ``candidates``?

    ``deriv_index`` is the lower index μ that contracts each candidate current's
    free upper index (so ``∂_μ J^μ`` is a scalar). Searching the candidate
    currents is the learnable step (Direction C), now in spacetime.
    """
    from indexcalc.core.deriv import PartialDeriv
    fields = frozenset(field_names)
    delta_L = div_expand(delta_L, fields)
    if _is_zero(delta_L):                       # δL = 0 exactly (internal sym)
        return True, None
    for J in (candidates or []):
        div_J = div_expand(PartialDeriv(J, deriv_index), fields)
        residual = TensorSum(delta_L, ScalarMul(-1.0, div_J))
        if _is_zero(residual):
            return True, J
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


def verify_symmetry_ft(
    lagrangian: TensorExpr,
    generator: Generator,
    field_names: Iterable[str],
    deriv_index: "Index",
    current_candidates: Optional[Iterable[TensorExpr]] = None,
) -> SymmetryResult:
    """Field-theory (∂_μ) symmetry check: ``δL = ∂_μ J^μ`` (total divergence).

    Spacetime analogue of ``verify_symmetry``. ``generator`` may be a
    derivative-dependent action (e.g. δφ = a^ν ∂_ν φ for spacetime translation);
    ``apply_generator`` already lifts ∂_μ through it. ``deriv_index`` is the
    lower index contracting each candidate current's free upper index.
    """
    delta_L = apply_generator(lagrangian, generator)
    ok, J = is_total_divergence(delta_L, field_names, deriv_index, current_candidates)
    return SymmetryResult(is_symmetry=ok, exact=(ok and J is None),
                          boundary_term=J, delta_L=delta_L)
