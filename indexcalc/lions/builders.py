"""Invariant tensor builders — metadata → concrete ``Tensor`` instances
that match the conventions used by IndexCalc simplifiers.

Each builder accepts the same convention pattern the corresponding
acceptance test uses, so the enumerator output is verifiable by the
existing oracle without per-shape adapter code.
"""

from __future__ import annotations

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor
from indexcalc.core.deriv import PartialDeriv, partial


def make_eta(spacetime: IndexSpace, mu: str = "μ", nu: str = "ν") -> Tensor:
    """η_{μν} — Minkowski metric, both slots lower, symmetric."""
    return Tensor(
        "eta",
        [spacetime.lower(mu), spacetime.lower(nu)],
        symmetric_pairs=[(0, 1)],
        reps={},
    )


def make_kronecker(
    space: IndexSpace, upper: str = "i", lower: str = "j",
) -> Tensor:
    """δ^i{}_j — Kronecker delta on a given IndexSpace."""
    return Tensor(
        "delta",
        [space.upper(upper), space.lower(lower)],
        reps={},
    )


def make_epsilon_su2(
    fund_space: IndexSpace, i: str = "i", j: str = "j",
) -> Tensor:
    """ε_{ij} — SU(2) fundamental antisymmetric invariant (both slots lower).

    Same convention as ``test_m8_acceptance.make_epsilon_lower``.
    """
    return Tensor(
        "epsilon",
        [fund_space.lower(i), fund_space.lower(j)],
        antisymmetric_pairs=[(0, 1)],
        reps={},
    )


def make_partial(field: Tensor, spacetime: IndexSpace, mu: str = "μ") -> PartialDeriv:
    """∂_μ on the given field, picking lower spacetime index name μ."""
    return partial(field, spacetime.lower(mu))
