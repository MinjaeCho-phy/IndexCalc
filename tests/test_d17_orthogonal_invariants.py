"""D17: O(N)/SO(N) invariant tensor registration smoke.

검증 toy:
1. ``standard_o_n_invariants(3)`` produces δ_{ij} symmetric + δ^i_j mixed.
2. ``standard_so_n_invariants(3)`` produces O(3) entries + ε_{ijk} totally antisymmetric.
3. ε는 SO(N) 한정 — O(N)에는 등록되지 않는다.
4. Registry에 넣고 is_invariant lookup이 작동.
5. Group dim: O(N) = SO(N) = N(N-1)/2, U(N) = N², SU(N) = N²-1.
"""

from __future__ import annotations

import pytest

from indexcalc.core.invariant_tensors import (
    InvariantTensor, InvariantTensorRegistry,
    standard_o_n_invariants, standard_so_n_invariants,
    standard_su_n_invariants, standard_u_n_invariants,
)


def test_o_n_has_symmetric_delta_and_mixed_identity():
    inv = standard_o_n_invariants(3)
    names = {t.name for t in inv}
    assert names == {"delta", "delta_mixed"}
    sym_delta = next(t for t in inv if t.name == "delta")
    assert sym_delta.symmetry == "symmetric"
    assert sym_delta.index_pattern == ("vector_lower", "vector_lower")
    mixed = next(t for t in inv if t.name == "delta_mixed")
    assert mixed.symmetry is None
    assert mixed.index_pattern == ("vector_upper", "vector_lower")


def test_o_n_does_not_include_epsilon():
    inv = standard_o_n_invariants(3)
    assert all(t.name != "epsilon" for t in inv)


def test_so_n_includes_epsilon_with_N_slots():
    inv = standard_so_n_invariants(3)
    eps = next(t for t in inv if t.name == "epsilon")
    assert eps.symmetry == "totally_antisymmetric"
    assert eps.index_pattern == ("vector_lower",) * 3
    inv5 = standard_so_n_invariants(5)
    eps5 = next(t for t in inv5 if t.name == "epsilon")
    assert eps5.index_pattern == ("vector_lower",) * 5


def test_so_n_inherits_o_n_invariants():
    so_inv = {t.name for t in standard_so_n_invariants(3)}
    o_inv = {t.name for t in standard_o_n_invariants(3)}
    assert o_inv.issubset(so_inv)


def test_registry_round_trip_for_o_so_groups():
    reg = InvariantTensorRegistry()
    for t in standard_o_n_invariants(3):
        reg.declare(t.name, t.group_name, t.index_pattern, t.symmetry)
    for t in standard_so_n_invariants(3):
        # delta는 O(3)와 SO(3) 양쪽에 등록 (group_name이 다르므로 key 충돌 없음)
        reg.declare(t.name, t.group_name, t.index_pattern, t.symmetry)

    assert reg.is_invariant("delta", "O(3)")
    assert reg.is_invariant("delta", "SO(3)")
    assert reg.is_invariant("epsilon", "SO(3)")
    assert not reg.is_invariant("epsilon", "O(3)")  # 핵심 차이
    assert sorted(reg.list_for_group("O(3)")) == ["delta", "delta_mixed"]
    assert sorted(reg.list_for_group("SO(3)")) == [
        "delta", "delta_mixed", "epsilon",
    ]


def test_group_dim_formulas():
    """그룹 dimension = parameter 수. probe 출력의 dim 필드용."""
    def o_n_dim(N): return N * (N - 1) // 2
    def so_n_dim(N): return N * (N - 1) // 2
    def u_n_dim(N): return N * N
    def su_n_dim(N): return N * N - 1
    assert o_n_dim(3) == 3
    assert so_n_dim(3) == 3
    assert u_n_dim(2) == 4
    assert su_n_dim(2) == 3


def test_existing_groups_unaffected():
    """기존 SU(N)/U(N)/Lorentz helper 정상 동작 확인 (regression guard)."""
    su2 = standard_su_n_invariants(2)
    assert {t.name for t in su2} == {"delta", "f", "d", "epsilon"}
    assert next(t for t in su2 if t.name == "epsilon").index_pattern == (
        "fund_lower", "fund_lower"
    )
    u2 = standard_u_n_invariants(2)
    assert "epsilon" not in {t.name for t in u2}
