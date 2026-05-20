"""v3 S1: Sp(2N) backend — generator + symplectic invariant tensor.

검증 toy (catalog 확장 v3.0):
1. ``standard_sp_2n_invariants(N)`` produces antisymmetric Ω_{ij} + mixed Ω^i_j.
2. Ω의 반대칭성이 δ (O(N)의 대칭)와 명확히 구분된다.
3. ``make_sp_2n_generator`` 가 vector + singlet action을 단다.
4. abelian group은 거부.
5. Group dim 공식: Sp(2N) = N(2N+1).
6. Registry round-trip — Ω가 Sp 그룹에 등록되고 lookup된다.
"""

from __future__ import annotations

import pytest

from indexcalc.core.group import Group
from indexcalc.core.index import IndexSpace
from indexcalc.core.generator import make_sp_2n_generator
from indexcalc.core.invariant_tensors import (
    InvariantTensorRegistry,
    standard_sp_2n_invariants,
    standard_o_n_invariants,
)


def test_sp_invariants_omega_is_antisymmetric():
    inv = standard_sp_2n_invariants(2)  # Sp(4)
    names = {t.name for t in inv}
    assert names == {"omega", "omega_mixed"}
    omega = next(t for t in inv if t.name == "omega")
    assert omega.symmetry == "antisymmetric"
    assert omega.index_pattern == ("vector_lower", "vector_lower")
    assert omega.group_name == "Sp(4)"
    mixed = next(t for t in inv if t.name == "omega_mixed")
    assert mixed.symmetry is None
    assert mixed.index_pattern == ("vector_upper", "vector_lower")


def test_sp_omega_distinct_from_o_n_delta():
    """핵심 차이: O(N) δ는 대칭, Sp(2N) Ω는 반대칭."""
    o_delta = next(t for t in standard_o_n_invariants(4) if t.name == "delta")
    sp_omega = next(t for t in standard_sp_2n_invariants(2) if t.name == "omega")
    assert o_delta.symmetry == "symmetric"
    assert sp_omega.symmetry == "antisymmetric"


def test_sp_group_name_uses_2n_dimension():
    """rank N → concrete label Sp(2N)."""
    assert standard_sp_2n_invariants(2)[0].group_name == "Sp(4)"
    assert standard_sp_2n_invariants(3)[0].group_name == "Sp(6)"
    assert standard_sp_2n_invariants(5)[0].group_name == "Sp(10)"


def test_make_sp_2n_generator_has_vector_and_singlet():
    sp4 = Group("Sp(4)", dim=10, abelian=False)
    sp4.add_rep("vector", dim=4)
    sp4.add_rep("singlet", dim=1)
    vec = IndexSpace("sp4_vec", dim=4, indices="ijklmn", metric="omega")
    g = make_sp_2n_generator(sp4, vec)
    assert g.has_action("vector")
    assert g.has_action("singlet")


def test_make_sp_2n_generator_rejects_abelian():
    ab = Group("Sp_bad", dim=1, abelian=True)
    vec = IndexSpace("bad_vec", dim=2, indices="ij", metric="omega")
    with pytest.raises(ValueError, match="non-abelian"):
        make_sp_2n_generator(ab, vec)


def test_sp_2n_dim_formula():
    """Sp(2N) dimension = N(2N+1). probe 출력의 dim 필드용."""
    def sp_dim(N): return N * (2 * N + 1)
    assert sp_dim(1) == 3   # Sp(2) ≅ SU(2), dim 3
    assert sp_dim(2) == 10  # Sp(4) ≅ Spin(5), dim 10
    assert sp_dim(3) == 21  # Sp(6)


def test_registry_round_trip_for_sp():
    reg = InvariantTensorRegistry()
    for t in standard_sp_2n_invariants(2):
        reg.declare(t.name, t.group_name, t.index_pattern, t.symmetry)
    assert reg.is_invariant("omega", "Sp(4)")
    assert reg.get("omega", "Sp(4)").symmetry == "antisymmetric"
    assert sorted(reg.list_for_group("Sp(4)")) == ["omega", "omega_mixed"]
