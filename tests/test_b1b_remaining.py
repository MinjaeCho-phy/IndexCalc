"""B1.b 잔여: LieDeriv + slice_decompose 회귀.

검증:
    - LieDeriv 기본 (scalar, vector, tensor) free_indices.
    - expand_lie_deriv가 textbook 형태 분해.
        - L_X φ = X^ρ ∂_ρ φ
        - L_X V^a = X^ρ ∂_ρ V^a - V^ρ ∂_ρ X^a
        - L_X T_{ab} = X^ρ ∂_ρ T_{ab} + T_{ρb} ∂_a X^ρ + T_{aρ} ∂_b X^ρ
    - K_evolution_rhs(include_shift_advection=True)에 LieDeriv 포함.
    - slice_decompose: rank-1, rank-2 키 구조와 free_indices.
    - LieDeriv 안의 Tensor 속성 보존 (antisym 등).
"""

import pytest

from indexcalc import (
    IndexSpace, Tensor, LeviCivitaConnection,
    ADMSetup, LieDeriv, expand_lie_deriv, slice_decompose,
    K_evolution_rhs,
    TensorSum, TensorProduct, ScalarMul,
    PartialDeriv,
)


@pytest.fixture
def st():
    return IndexSpace("st", dim=4, indices="μνρσ", metric="g")


@pytest.fixture
def sp():
    return IndexSpace("sp", dim=3, indices="ijklmn", metric="h")


@pytest.fixture
def adm(st, sp):
    return ADMSetup(st, sp)


@pytest.fixture
def conn3(adm):
    return LeviCivitaConnection(
        adm.spatial_metric_lower(), adm.spatial_metric_upper(), adm.sp,
    )


# ─── LieDeriv basics ───────────────────────────────────────


class TestLieDerivBasics:
    def test_vector_must_be_upper_one_index(self, st):
        # 0-index → 에러
        bad = Tensor("X", [])
        T = Tensor("T", [])
        with pytest.raises(ValueError, match="1-index"):
            LieDeriv(bad, T)
        # lower index → 에러
        X_lo = Tensor("X", [st.lower("μ")])
        with pytest.raises(ValueError, match="upper"):
            LieDeriv(X_lo, T)

    def test_free_indices_preserved(self, st):
        X = Tensor("X", [st.upper("μ")])
        T = Tensor("T", [st.upper("a"), st.lower("b")])
        Lie = LieDeriv(X, T)
        assert [i.name for i in Lie.free_indices] == ["a", "b"]


# ─── expand_lie_deriv ─────────────────────────────────────


class TestExpandLie:
    def test_scalar_only_advection(self, st):
        """L_X φ = X^ρ ∂_ρ φ — 1 항만 (φ에 인덱스 없으니 보정 없음)."""
        X = Tensor("X", [st.upper("μ")])
        phi = Tensor("φ", [])
        out = expand_lie_deriv(LieDeriv(X, phi))
        assert isinstance(out, TensorProduct)
        assert out.left.name == "X"
        assert isinstance(out.right, PartialDeriv)
        assert out.right.expr.name == "φ"

    def test_vector_two_terms(self, st):
        """L_X V^a → 2 항 (advection - V^ρ ∂_ρ X^a)."""
        X = Tensor("X", [st.upper("μ")])
        V = Tensor("V", [st.upper("a")])
        out = expand_lie_deriv(LieDeriv(X, V))
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(out)
        assert len(terms) == 2
        # 두 번째 항은 -X
        assert isinstance(terms[1], ScalarMul)
        assert terms[1].scalar == -1

    def test_lower_tensor_three_terms(self, st):
        """L_X T_{ab} → 3 항 (advection + T_{ρb} ∂_a X^ρ + T_{aρ} ∂_b X^ρ)."""
        X = Tensor("X", [st.upper("μ")])
        T = Tensor("T", [st.lower("a"), st.lower("b")])
        out = expand_lie_deriv(LieDeriv(X, T))
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(out)
        assert len(terms) == 3
        # 모두 + 부호 (advection + 2 lower 보정)
        for t in terms[1:]:
            assert not isinstance(t, ScalarMul) or t.scalar > 0

    def test_attributes_preserved(self, st):
        """LieDeriv 보정 항의 T_replaced에 antisymmetric_pairs 보존."""
        X = Tensor("X", [st.upper("μ")])
        F = Tensor(
            "F", [st.lower("a"), st.lower("b")],
            antisymmetric_pairs=[(0, 1)],
        )
        out = expand_lie_deriv(LieDeriv(X, F))
        from indexcalc.core.simplify import _flatten_sum
        for term in _flatten_sum(out):
            current = term
            while isinstance(current, ScalarMul):
                current = current.expr
            if isinstance(current, TensorProduct):
                # 보정 항에 F (replaced) 포함
                from indexcalc.core.simplify import collect_factors
                for f in collect_factors(current):
                    if isinstance(f, Tensor) and f.name == "F":
                        assert f.antisymmetric_pairs == ((0, 1),)


# ─── K_evolution shift advection 옵션 ───────────────────


class TestKEvolutionShiftAdvection:
    def test_default_no_lie(self, adm, conn3):
        rhs = K_evolution_rhs(adm, conn3)
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(rhs)
        # default 4 항
        assert len(terms) == 4
        # LieDeriv 없음
        assert all(not isinstance(t, LieDeriv) for t in terms)

    def test_with_shift_advection(self, adm, conn3):
        rhs = K_evolution_rhs(adm, conn3, include_shift_advection=True)
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(rhs)
        assert len(terms) == 5
        assert any(isinstance(t, LieDeriv) for t in terms)


# ─── slice_decompose ─────────────────────────────────────


class TestSliceDecompose:
    def test_rank0_single_entry(self, st, sp):
        T = Tensor("T", [])
        out = slice_decompose(T, sp)
        assert set(out.keys()) == {()}
        assert out[()].indices == ()

    def test_rank1_two_entries(self, st, sp):
        T = Tensor("T", [st.upper("μ")])
        out = slice_decompose(T, sp)
        assert set(out.keys()) == {("t",), ("i",)}
        assert out[("t",)].indices == ()
        assert len(out[("i",)].indices) == 1

    def test_rank2_four_entries(self, st, sp):
        T = Tensor("T", [st.lower("μ"), st.lower("ν")])
        out = slice_decompose(T, sp)
        assert set(out.keys()) == {("t", "t"), ("t", "i"), ("i", "t"), ("i", "j")}
        assert out[("t", "t")].indices == ()
        assert len(out[("t", "i")].indices) == 1
        assert len(out[("i", "j")].indices) == 2

    def test_position_preserved(self, st, sp):
        T = Tensor("T", [st.upper("μ"), st.lower("ν")])
        out = slice_decompose(T, sp)
        # ('i','j') 항: 첫 spatial은 upper, 둘째는 lower
        comp = out[("i", "j")]
        assert comp.indices[0].position == "upper"
        assert comp.indices[1].position == "lower"

    def test_name_suffix(self, st, sp):
        T = Tensor("T", [st.lower("μ"), st.lower("ν")])
        out = slice_decompose(T, sp)
        assert out[("t", "t")].name == "T_tt"
        assert out[("i", "j")].name == "T_ij"

    def test_no_suffix_option(self, st, sp):
        T = Tensor("T", [st.lower("μ")])
        out = slice_decompose(T, sp, name_suffix=False)
        assert out[("t",)].name == "T"
        assert out[("i",)].name == "T"
