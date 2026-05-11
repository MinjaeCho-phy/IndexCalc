"""Backend 1.b: ADM 3+1 분해 + extrinsic curvature 회귀 테스트.

검증 항목:
    - ADMSetup이 4D/3D 차원 강제, leaf builder가 옳은 Tensor 반환.
    - TimeDeriv가 free index에 t를 추가하지 않음.
    - extrinsic_curvature_definition의 free=[i,j], 구조에 TimeDeriv +
      SpatialCovariantDeriv가 등장.
    - K_trace_definition의 free=[].
    - metric_lower/upper_components의 'tt' 'ti' 'ij' 항이 옳은 free count.
"""

import pytest
import sympy as sp

from indexcalc import (
    IndexSpace, Tensor, MetricRegistry, LeviCivitaConnection,
    TensorSum, TensorProduct, ScalarMul,
    ADMSetup, TimeDeriv,
    extrinsic_curvature_definition, K_trace_definition,
    metric_lower_components, metric_upper_components,
)
from indexcalc.core.spatial_deriv import SpatialCovariantDeriv


@pytest.fixture
def st():
    return IndexSpace("st", dim=4, indices="μνρσ", metric="g")


@pytest.fixture
def sp_():
    return IndexSpace("sp", dim=3, indices="ijklmn", metric="h")


@pytest.fixture
def adm(st, sp_):
    return ADMSetup(st, sp_)


@pytest.fixture
def conn3(adm):
    h_lo = adm.spatial_metric_lower()
    h_up = adm.spatial_metric_upper()
    return LeviCivitaConnection(h_lo, h_up, adm.sp)


# ─── ADMSetup dimension guards ─────────────────────────────


class TestADMSetup:
    def test_rejects_non_4d_st(self, sp_):
        bad_st = IndexSpace("bad", dim=3, indices="abc")
        with pytest.raises(ValueError, match="4D spacetime"):
            ADMSetup(bad_st, sp_)

    def test_rejects_non_3d_sp(self, st):
        bad_sp = IndexSpace("bad", dim=2, indices="ab")
        with pytest.raises(ValueError, match="3D spatial"):
            ADMSetup(st, bad_sp)

    def test_lapse_is_scalar(self, adm):
        N = adm.lapse()
        assert N.indices == ()
        assert N.free_indices == []

    def test_shift_default_upper(self, adm):
        N_up = adm.shift()
        assert len(N_up.indices) == 1
        assert N_up.indices[0].position == "upper"

    def test_shift_lower(self, adm):
        N_lo = adm.shift("i", "lower")
        assert N_lo.indices[0].position == "lower"

    def test_spatial_metric_symmetric(self, adm):
        h = adm.spatial_metric_lower()
        assert h.symmetric_pairs == ((0, 1),)
        assert all(i.position == "lower" for i in h.indices)

    def test_spatial_metric_upper_symmetric(self, adm):
        h_inv = adm.spatial_metric_upper()
        assert h_inv.symmetric_pairs == ((0, 1),)
        assert all(i.position == "upper" for i in h_inv.indices)

    def test_extrinsic_curvature_symmetric(self, adm):
        K = adm.extrinsic_curvature()
        assert K.symmetric_pairs == ((0, 1),)
        assert K.indices[0].position == "lower"
        assert K.indices[1].position == "lower"


# ─── TimeDeriv ─────────────────────────────────────────────


class TestTimeDeriv:
    def test_no_extra_free_index(self, adm):
        h = adm.spatial_metric_lower()
        dt_h = TimeDeriv(h)
        names = [i.name for i in dt_h.free_indices]
        assert names == ["i", "j"]
        assert "t" not in names

    def test_preserves_inner_free(self, adm):
        N = adm.shift("i", "lower")
        dt_N = TimeDeriv(N)
        assert [i.name for i in dt_N.free_indices] == ["i"]


# ─── extrinsic curvature definition ─────────────────────────


class TestExtrinsicCurvatureDefinition:
    def test_free_indices(self, adm, conn3):
        K_def = extrinsic_curvature_definition(adm, conn3)
        free = sorted(i.name for i in K_def.free_indices)
        assert free == ["i", "j"]

    def test_outer_is_scalar_mul(self, adm, conn3):
        K_def = extrinsic_curvature_definition(adm, conn3)
        assert isinstance(K_def, ScalarMul)
        # coefficient = 1/(2N)
        N_sym = sp.Symbol("N")
        assert sp.simplify(K_def.scalar - 1 / (2 * N_sym)) == 0

    def test_inner_contains_time_deriv_and_spatial_deriv(self, adm, conn3):
        K_def = extrinsic_curvature_definition(adm, conn3)
        # Walk inner sum, find at least one TimeDeriv and one SpatialCovariantDeriv
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(K_def.expr)
        assert len(terms) == 3
        # First term ∂_t h
        # 두 번째/세 번째: -D_i N_j, -D_j N_i (ScalarMul wrappers)
        types_seen = set()
        for term in terms:
            current = term
            while isinstance(current, ScalarMul):
                current = current.expr
            types_seen.add(type(current).__name__)
        assert "TimeDeriv" in types_seen
        assert "SpatialCovariantDeriv" in types_seen


# ─── K trace ───────────────────────────────────────────────


class TestKTrace:
    def test_scalar_free(self, adm):
        K_tr = K_trace_definition(adm)
        assert K_tr.free_indices == []

    def test_structure_h_inv_times_K(self, adm):
        K_tr = K_trace_definition(adm)
        assert isinstance(K_tr, TensorProduct)
        assert K_tr.left.name == "h"
        assert K_tr.right.name == "K"


# ─── metric components ─────────────────────────────────────


class TestMetricComponents:
    def test_lower_keys(self, adm):
        comps = metric_lower_components(adm)
        assert set(comps) == {"tt", "ti", "ij"}

    def test_lower_free_counts(self, adm):
        comps = metric_lower_components(adm)
        assert len(comps["tt"].free_indices) == 0
        assert len(comps["ti"].free_indices) == 1
        assert len(comps["ij"].free_indices) == 2

    def test_upper_keys(self, adm):
        comps = metric_upper_components(adm)
        assert set(comps) == {"tt", "ti", "ij"}

    def test_upper_free_counts(self, adm):
        comps = metric_upper_components(adm)
        assert len(comps["tt"].free_indices) == 0
        assert len(comps["ti"].free_indices) == 1
        assert len(comps["ij"].free_indices) == 2

    def test_upper_tt_is_negative_inv_N_sq(self, adm):
        """g^tt = -1/N^2 (ScalarMul wrapping)."""
        g_inv_tt = metric_upper_components(adm)["tt"]
        assert isinstance(g_inv_tt, ScalarMul)
        N_sym = sp.Symbol("N")
        assert sp.simplify(g_inv_tt.scalar - (-1 / N_sym ** 2)) == 0

    def test_lower_tt_includes_shift_dot_shift(self, adm):
        """g_tt = -N^2 + N_k N^k. 두 항 합."""
        g_tt = metric_lower_components(adm)["tt"]
        # 합 구조: TensorSum(-N⊗N, N_k⊗N^k)
        assert isinstance(g_tt, TensorSum)

    def test_lower_ij_is_h(self, adm):
        g_ij = metric_lower_components(adm)["ij"]
        assert isinstance(g_ij, Tensor)
        assert g_ij.name == "h"
