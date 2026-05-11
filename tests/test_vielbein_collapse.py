"""Vielbein identity collapse 회귀.

규칙:
    - e^a_μ η_{ab} e^b_ν → g_{μν}  (lower 버전)
    - e_a^μ η^{ab} e_b^ν → g^{μν}  (upper 버전)
    - 다른 factor와 곱: T_ρσ × (e η e) → T_ρσ × g
    - 패턴 없으면 변경 없음
    - free index 보존
    - 다중 identity 연속 collapse
"""

import pytest

from indexcalc import (
    IndexSpace, Tensor, TensorProduct, TensorSum, ScalarMul,
    VielbeinRegistry, collapse_vielbein_identity,
)
from indexcalc.core.index import Index


@pytest.fixture
def st():
    return IndexSpace("st", dim=4, indices="μνρσλ", metric="g")


@pytest.fixture
def fr():
    return IndexSpace("fr", dim=4, indices="abcde", metric="η")


@pytest.fixture
def vbreg(st, fr):
    r = VielbeinRegistry()
    r.register("e", "η", "g", fr, st)
    return r


def _e_lower(fr, st, a, μ):
    """e^a{}_μ — frame upper, st lower."""
    return Tensor("e", [Index(a, fr, "upper"), Index(μ, st, "lower")])


def _e_upper(fr, st, a, μ):
    """e_a{}^μ — frame lower, st upper."""
    return Tensor("e", [Index(a, fr, "lower"), Index(μ, st, "upper")])


def _eta_lower(fr, a, b):
    return Tensor(
        "η", [Index(a, fr, "lower"), Index(b, fr, "lower")],
        symmetric_pairs=[(0, 1)],
    )


def _eta_upper(fr, a, b):
    return Tensor(
        "η", [Index(a, fr, "upper"), Index(b, fr, "upper")],
        symmetric_pairs=[(0, 1)],
    )


# ─── Basic collapse ───────────────────────────────────────


class TestBasicCollapse:
    def test_lower_pattern(self, st, fr, vbreg):
        """e^a_μ η_{ab} e^b_ν → g_{μν}."""
        e1 = _e_lower(fr, st, "a", "μ")
        eta = _eta_lower(fr, "a", "b")
        e2 = _e_lower(fr, st, "b", "ν")
        expr = TensorProduct(TensorProduct(e1, eta), e2)
        result = collapse_vielbein_identity(expr, vbreg)
        assert isinstance(result, Tensor)
        assert result.name == "g"
        assert sorted(i.name for i in result.indices) == ["μ", "ν"]
        assert all(i.position == "lower" for i in result.indices)

    def test_upper_pattern(self, st, fr, vbreg):
        """e_a^μ η^{ab} e_b^ν → g^{μν}."""
        e1 = _e_upper(fr, st, "a", "μ")
        eta = _eta_upper(fr, "a", "b")
        e2 = _e_upper(fr, st, "b", "ν")
        expr = TensorProduct(TensorProduct(e1, eta), e2)
        result = collapse_vielbein_identity(expr, vbreg)
        assert isinstance(result, Tensor)
        assert result.name == "g"
        assert all(i.position == "upper" for i in result.indices)

    def test_free_indices_preserved(self, st, fr, vbreg):
        """collapse 전후 free index 일치 (μ, ν 보존)."""
        e1 = _e_lower(fr, st, "a", "μ")
        eta = _eta_lower(fr, "a", "b")
        e2 = _e_lower(fr, st, "b", "ν")
        expr = TensorProduct(TensorProduct(e1, eta), e2)
        before = sorted(i.name for i in expr.free_indices)
        result = collapse_vielbein_identity(expr, vbreg)
        after = sorted(i.name for i in result.free_indices)
        assert before == after


# ─── With other factors ──────────────────────────────────


class TestWithOtherFactors:
    def test_extra_factor_preserved(self, st, fr, vbreg):
        """T_ρσ × (e η e) → T_ρσ × g."""
        T = Tensor("T", [Index("ρ", st, "lower"), Index("σ", st, "lower")])
        e1 = _e_lower(fr, st, "a", "μ")
        eta = _eta_lower(fr, "a", "b")
        e2 = _e_lower(fr, st, "b", "ν")
        expr = TensorProduct(T, TensorProduct(TensorProduct(e1, eta), e2))
        result = collapse_vielbein_identity(expr, vbreg)
        assert isinstance(result, TensorProduct)
        # T와 g 둘 다 있음
        from indexcalc.core.simplify import collect_factors
        facs = collect_factors(result)
        names = sorted(f.name for f in facs if isinstance(f, Tensor))
        assert names == ["T", "g"]


# ─── No match → unchanged ────────────────────────────────


class TestNoMatch:
    def test_no_eta_no_collapse(self, st, fr, vbreg):
        """e^a_μ × e^b_ν 만 있고 η 없음 → 변경 없음."""
        e1 = _e_lower(fr, st, "a", "μ")
        e2 = _e_lower(fr, st, "b", "ν")
        expr = TensorProduct(e1, e2)
        result = collapse_vielbein_identity(expr, vbreg)
        assert result is expr

    def test_unrelated_tensors_unchanged(self, st, fr, vbreg):
        """Vielbein과 무관한 tensor 곱은 그대로."""
        A = Tensor("A", [Index("μ", st, "upper")])
        B = Tensor("B", [Index("μ", st, "lower")])
        expr = TensorProduct(A, B)
        result = collapse_vielbein_identity(expr, vbreg)
        assert result is expr

    def test_wrong_position_no_collapse(self, st, fr, vbreg):
        """e의 frame upper인데 η도 upper면 contract 안 됨 → collapse 안 함."""
        e1 = _e_lower(fr, st, "a", "μ")  # frame upper
        # η를 upper로 (잘못된 contraction)
        eta_up = _eta_upper(fr, "a", "b")
        e2 = _e_lower(fr, st, "b", "ν")
        # 이 product는 a가 e1 upper + eta upper = 같은 위치 → Einstein 위반.
        # TensorProduct가 만들어질 때 'a' 3+회 카운트는 아니지만 같은 위치 contract 시도 X.
        # Einstein 위반 자체는 TensorProduct 생성에선 검사 안 됨 (validate_einstein은 별도).
        # 어쨌든 collapse_vielbein_identity는 position check로 reject.
        expr = TensorProduct(TensorProduct(e1, eta_up), e2)
        result = collapse_vielbein_identity(expr, vbreg)
        # 패턴 매치 실패 → 그대로
        assert result is expr


# ─── Recursive in Sum/ScalarMul ──────────────────────────


class TestRecursive:
    def test_inside_sum(self, st, fr, vbreg):
        """(e η e) + X → g + X."""
        e1 = _e_lower(fr, st, "a", "μ")
        eta = _eta_lower(fr, "a", "b")
        e2 = _e_lower(fr, st, "b", "ν")
        triple = TensorProduct(TensorProduct(e1, eta), e2)
        X = Tensor("X", [Index("μ", st, "lower"), Index("ν", st, "lower")])
        expr = TensorSum(triple, X)
        result = collapse_vielbein_identity(expr, vbreg)
        assert isinstance(result, TensorSum)
        # 왼쪽이 g가 되었는지
        assert isinstance(result.left, Tensor)
        assert result.left.name == "g"

    def test_inside_scalar_mul(self, st, fr, vbreg):
        e1 = _e_lower(fr, st, "a", "μ")
        eta = _eta_lower(fr, "a", "b")
        e2 = _e_lower(fr, st, "b", "ν")
        triple = TensorProduct(TensorProduct(e1, eta), e2)
        expr = ScalarMul(2, triple)
        result = collapse_vielbein_identity(expr, vbreg)
        assert isinstance(result, ScalarMul)
        assert result.scalar == 2
        assert isinstance(result.expr, Tensor)
        assert result.expr.name == "g"


# ─── Chain collapse ──────────────────────────────────────


class TestChainCollapse:
    def test_two_identities_in_one_product(self, st, fr, vbreg):
        """(e η e)_{μν} × (e η e)_{ρσ} → g_{μν} × g_{ρσ} 한 번의 호출에 모두 처리."""
        e1 = _e_lower(fr, st, "a", "μ")
        eta1 = _eta_lower(fr, "a", "b")
        e2 = _e_lower(fr, st, "b", "ν")
        e3 = _e_lower(fr, st, "c", "ρ")
        eta2 = _eta_lower(fr, "c", "d")
        e4 = _e_lower(fr, st, "d", "σ")
        big = TensorProduct(
            TensorProduct(TensorProduct(e1, eta1), e2),
            TensorProduct(TensorProduct(e3, eta2), e4),
        )
        result = collapse_vielbein_identity(big, vbreg)
        # 모든 factor가 g가 되어야 함 (정확히 2개 g)
        from indexcalc.core.simplify import collect_factors
        facs = collect_factors(result)
        g_count = sum(1 for f in facs if isinstance(f, Tensor) and f.name == "g")
        assert g_count == 2
