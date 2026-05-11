"""G8: free_indices의 Einstein 자동 contraction 회귀 테스트.

규칙:
    - Tensor가 같은 이름·반대 위치 인덱스를 정확히 두 번 가지면 그 쌍은
      free_indices에서 제외된다 (self-trace).
    - PartialDeriv/CovariantDeriv의 deriv_index가 inner의 free 인덱스와
      같은 이름·반대 위치이면 contract된 것으로 보고 free에서 제거.
    - Ricci tensor를 직접 텐서 곱/합으로 구성할 수 있다 (Step 5 unblock).
"""

import pytest

from indexcalc import (
    IndexSpace, Tensor, TensorProduct, TensorSum,
    PartialDeriv, partial,
    CovariantDeriv, LeviCivitaConnection, MetricRegistry,
)
from indexcalc.core.index import Index


@pytest.fixture
def st():
    return IndexSpace("st", dim=4, indices="μνρσλαβ", metric="g")


# ─── Tensor self-contraction ───────────────────────────────


class TestTensorSelfContract:
    def test_self_paired_indices_become_dummies(self, st):
        """Γ^ρ_ρλ → free=[λ]."""
        ρU = Index("ρ", st, "upper")
        ρL = Index("ρ", st, "lower")
        λL = Index("λ", st, "lower")
        Γ = Tensor("Γ", [ρU, ρL, λL])
        free = Γ.free_indices
        assert len(free) == 1
        assert free[0].name == "λ"
        assert free[0].position == "lower"

    def test_no_self_pair_unchanged(self, st):
        """일반 Γ^ρ_νλ → free=[ρ, ν, λ]."""
        ρU = Index("ρ", st, "upper")
        νL = Index("ν", st, "lower")
        λL = Index("λ", st, "lower")
        Γ = Tensor("Γ", [ρU, νL, λL])
        free = Γ.free_indices
        assert len(free) == 3
        assert [i.name for i in free] == ["ρ", "ν", "λ"]

    def test_same_name_same_position_not_contracted(self, st):
        """T_μ_μ (둘 다 lower) — Einstein 위반이지만 free=[μ, μ] 그대로."""
        μL1 = Index("μ", st, "lower")
        μL2 = Index("μ", st, "lower")
        T = Tensor("T", [μL1, μL2])
        free = T.free_indices
        assert len(free) == 2

    def test_rank_uses_contracted_free(self, st):
        """Γ^ρ_ρλ.rank == (0, 1)."""
        ρU = Index("ρ", st, "upper")
        ρL = Index("ρ", st, "lower")
        λL = Index("λ", st, "lower")
        Γ = Tensor("Γ", [ρU, ρL, λL])
        assert Γ.rank == (0, 1)

    def test_indices_attribute_preserved(self, st):
        """Tensor.indices는 원본 그대로 유지 (self-contract 영향 없음)."""
        ρU = Index("ρ", st, "upper")
        ρL = Index("ρ", st, "lower")
        λL = Index("λ", st, "lower")
        Γ = Tensor("Γ", [ρU, ρL, λL])
        assert len(Γ.indices) == 3
        assert [i.name for i in Γ.indices] == ["ρ", "ρ", "λ"]


# ─── PartialDeriv: deriv_index ↔ inner contract ──────────


class TestPartialDerivContract:
    def test_partial_contracts_with_inner_free(self, st):
        """∂_ρ Γ^ρ_νσ → free=[ν, σ]."""
        ρU = Index("ρ", st, "upper")
        νL = Index("ν", st, "lower")
        σL = Index("σ", st, "lower")
        ρL = Index("ρ", st, "lower")
        Γ = Tensor("Γ", [ρU, νL, σL])
        d = PartialDeriv(Γ, ρL)
        free = d.free_indices
        assert len(free) == 2
        assert {i.name for i in free} == {"ν", "σ"}

    def test_partial_no_match_keeps_deriv_index_free(self, st):
        """∂_μ Γ^ρ_νσ → free=[μ, ρ, ν, σ] (μ는 inner와 무관)."""
        ρU = Index("ρ", st, "upper")
        νL = Index("ν", st, "lower")
        σL = Index("σ", st, "lower")
        μL = Index("μ", st, "lower")
        Γ = Tensor("Γ", [ρU, νL, σL])
        d = PartialDeriv(Γ, μL)
        free = d.free_indices
        assert len(free) == 4
        assert {i.name for i in free} == {"μ", "ρ", "ν", "σ"}

    def test_partial_on_self_contracted_inner(self, st):
        """∂_ν Γ^ρ_ρσ — inner already self-contracts (free=[σ]). deriv ν 추가 → free=[ν, σ]."""
        ρU = Index("ρ", st, "upper")
        ρL = Index("ρ", st, "lower")
        σL = Index("σ", st, "lower")
        νL = Index("ν", st, "lower")
        Γ = Tensor("Γ", [ρU, ρL, σL])
        d = PartialDeriv(Γ, νL)
        free = d.free_indices
        assert len(free) == 2
        assert {i.name for i in free} == {"ν", "σ"}


# ─── CovariantDeriv: deriv_index ↔ inner contract ───────


class TestCovariantDerivContract:
    def test_covariant_contracts_with_inner_free(self, st):
        """∇_ρ T^ρ_ν → free=[ν]."""
        mreg = MetricRegistry()
        g_lo = Tensor("g", [st.lower("μ"), st.lower("ν")])
        g_up = Tensor("g", [st.upper("μ"), st.upper("ν")])
        mreg.register(g_lo, g_up, st)
        conn = LeviCivitaConnection(g_lo, g_up, st)

        ρU = Index("ρ", st, "upper")
        νL = Index("ν", st, "lower")
        ρL = Index("ρ", st, "lower")
        T = Tensor("T", [ρU, νL])
        cd = CovariantDeriv(T, ρL, {st.name: conn})
        free = cd.free_indices
        assert len(free) == 1
        assert free[0].name == "ν"


# ─── Ricci tensor construction (Step 5) ─────────────────


class TestRicciConstruction:
    def test_ricci_definition_constructs(self, st):
        """R_σν = ∂_ρΓ^ρ_νσ − ∂_νΓ^ρ_ρσ + Γ^ρ_ρλ Γ^λ_νσ − Γ^ρ_νλ Γ^λ_ρσ.

        진단 스크립트 Step 5의 핵심: 모든 항이 free=[σ, ν]로 일치해야 TensorSum
        에러 없이 합쳐진다.
        """
        def Γ(up, lo1, lo2):
            return Tensor("Γ", [
                Index(up,  st, "upper"),
                Index(lo1, st, "lower"),
                Index(lo2, st, "lower"),
            ])

        ρ_lo = Index("ρ", st, "lower")
        ν_lo = Index("ν", st, "lower")

        term1 = PartialDeriv(Γ("ρ", "ν", "σ"), ρ_lo)        # ∂_ρ Γ^ρ_νσ
        term2 = PartialDeriv(Γ("ρ", "ρ", "σ"), ν_lo)         # ∂_ν Γ^ρ_ρσ
        quad1 = TensorProduct(Γ("ρ", "ρ", "λ"), Γ("λ", "ν", "σ"))
        quad2 = TensorProduct(Γ("ρ", "ν", "λ"), Γ("λ", "ρ", "σ"))

        # 각 항이 free=[σ, ν] 또는 [ν, σ]로 같은 free 카운트(2)
        assert len(term1.free_indices) == 2
        assert len(term2.free_indices) == 2
        assert len(quad1.free_indices) == 2
        assert len(quad2.free_indices) == 2

        # 모두 합쳐도 에러 없음
        ricci = (term1 - term2) + (quad1 - quad2)
        assert len(ricci.free_indices) == 2
        assert {i.name for i in ricci.free_indices} == {"σ", "ν"}
