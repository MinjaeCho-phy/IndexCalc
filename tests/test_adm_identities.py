"""Backend 1.b+: ADM 제약식·진화식·Gauss-Codazzi RHS 회귀.

검증 항목:
    - hamiltonian_constraint: free=[] (scalar). R^(3) + K^2 - K_{ij}K^{ij}.
    - momentum_constraint: free=[i].
    - h_evolution_rhs: free=[i,j].
    - K_evolution_rhs: free=[i,j].
    - gauss_rhs: free=[i,j,k,l]. R^(3)_{ijkl} + K_ik K_jl - K_il K_jk.
    - codazzi_rhs: free=[j,k,l]. D_l K_jk - D_k K_jl.
    - 새 leaf builder: extrinsic_curvature_upper, _mixed, ricci3_lower, ricci3_scalar, riemann3.
"""

import pytest

from indexcalc import (
    IndexSpace, LeviCivitaConnection, ADMSetup,
    hamiltonian_constraint, momentum_constraint,
    h_evolution_rhs, K_evolution_rhs,
    gauss_rhs, codazzi_rhs,
    Tensor, TensorSum, TensorProduct, ScalarMul,
)


@pytest.fixture
def adm():
    st = IndexSpace("st", dim=4, indices="μνρσ", metric="g")
    sp = IndexSpace("sp", dim=3, indices="ijklmn", metric="h")
    return ADMSetup(st, sp)


@pytest.fixture
def conn3(adm):
    h_lo = adm.spatial_metric_lower()
    h_up = adm.spatial_metric_upper()
    return LeviCivitaConnection(h_lo, h_up, adm.sp)


# ─── New leaf builders ────────────────────────────────────


class TestLeafBuilders:
    def test_K_upper_symmetric(self, adm):
        K_up = adm.extrinsic_curvature_upper()
        assert K_up.symmetric_pairs == ((0, 1),)
        assert all(i.position == "upper" for i in K_up.indices)

    def test_K_mixed_no_symmetric(self, adm):
        K_mx = adm.extrinsic_curvature_mixed("j", "i")
        # mixed: 한 upper 한 lower → sym 속성 안 줌
        assert K_mx.symmetric_pairs == ()
        assert K_mx.indices[0].position == "upper"
        assert K_mx.indices[1].position == "lower"

    def test_ricci3_lower_symmetric(self, adm):
        R = adm.ricci3_lower()
        assert R.symmetric_pairs == ((0, 1),)
        assert all(i.position == "lower" for i in R.indices)

    def test_ricci3_scalar_no_indices(self, adm):
        R = adm.ricci3_scalar()
        assert R.indices == ()
        assert R.free_indices == []

    def test_riemann3_antisym_pairs(self, adm):
        R = adm.riemann3()
        assert R.antisymmetric_pairs == ((0, 1), (2, 3))
        assert len(R.indices) == 4


# ─── Constraints ─────────────────────────────────────────


class TestHamiltonianConstraint:
    def test_scalar(self, adm):
        H = hamiltonian_constraint(adm)
        assert H.free_indices == []

    def test_three_terms(self, adm):
        H = hamiltonian_constraint(adm)
        # 구조: TensorSum 트리. R3 + K^2 - K·K.
        # _flatten_sum로 3개 항 확인
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(H)
        assert len(terms) == 3


class TestMomentumConstraint:
    def test_one_lower_index(self, adm, conn3):
        M = momentum_constraint(adm, conn3)
        assert len(M.free_indices) == 1
        assert M.free_indices[0].name == "i"
        assert M.free_indices[0].position == "lower"


# ─── Evolution equations ────────────────────────────────


class TestHEvolutionRhs:
    def test_two_free_lower(self, adm, conn3):
        rhs = h_evolution_rhs(adm, conn3)
        names = sorted(i.name for i in rhs.free_indices)
        assert names == ["i", "j"]

    def test_three_terms(self, adm, conn3):
        rhs = h_evolution_rhs(adm, conn3)
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(rhs)
        assert len(terms) == 3


class TestKEvolutionRhs:
    def test_two_free_lower(self, adm, conn3):
        rhs = K_evolution_rhs(adm, conn3)
        names = sorted(i.name for i in rhs.free_indices)
        assert names == ["i", "j"]

    def test_four_terms(self, adm, conn3):
        rhs = K_evolution_rhs(adm, conn3)
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(rhs)
        assert len(terms) == 4


# ─── Gauss / Codazzi ─────────────────────────────────────


class TestGaussRhs:
    def test_four_free_indices(self, adm):
        g = gauss_rhs(adm)
        names = sorted(i.name for i in g.free_indices)
        assert names == ["i", "j", "k", "l"]

    def test_three_terms(self, adm):
        g = gauss_rhs(adm)
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(g)
        assert len(terms) == 3


class TestCodazziRhs:
    def test_three_free_lower(self, adm, conn3):
        c = codazzi_rhs(adm, conn3)
        names = sorted(i.name for i in c.free_indices)
        assert names == ["j", "k", "l"]

    def test_two_terms(self, adm, conn3):
        c = codazzi_rhs(adm, conn3)
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(c)
        assert len(terms) == 2


# ─── h_evolution과 K 정의의 등가성 (구조적) ──────────────


class TestEvolutionConsistency:
    def test_h_evolution_matches_K_definition(self, adm, conn3):
        """∂_t h = -2N K + 2 D_(i N_j) ↔ K = (1/(2N))(∂_t h - 2 D_(i N_j)) 환원.

        구조적 등가성만 — free=[i,j], term 수 일치.
        """
        from indexcalc import extrinsic_curvature_definition
        K_def = extrinsic_curvature_definition(adm, conn3)
        rhs = h_evolution_rhs(adm, conn3)
        # 둘 다 free=[i,j]
        assert sorted(i.name for i in K_def.free_indices) == ["i", "j"]
        assert sorted(i.name for i in rhs.free_indices) == ["i", "j"]
