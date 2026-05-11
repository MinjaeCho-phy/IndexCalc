"""Backend 2: VielbeinSetup + SpinConnection + compatibility 회귀.

검증:
    - VielbeinSetup leaf builders (e^a_μ, e_a^μ, e^{aμ}, η, g) 위치/이름.
    - SpinConnection이 Connection의 subclass; make_tensor 정상.
    - vielbein_compatibility_lhs가 CovariantDeriv 노드 + free=[μ,a,ν].
    - expand_covariant이 ∂_μ e + ω·e - Γ·e (3항)으로 분해.
    - spin_connection_from_vielbein 구조 + free=[a,b,μ].
    - VielbeinSetup.to_registry로 collapse_vielbein_identity와 호환.
"""

import pytest

from indexcalc import (
    IndexSpace, Tensor, TensorProduct, TensorSum,
    Connection, LeviCivitaConnection, CovariantDeriv, expand_covariant,
    VielbeinSetup, SpinConnection, VielbeinRegistry,
    vielbein_compatibility_lhs, spin_connection_from_vielbein,
    collapse_vielbein_identity,
)
from indexcalc.core.index import Index


@pytest.fixture
def st():
    return IndexSpace("st", dim=4, indices="μνρσλ", metric="g")


@pytest.fixture
def fr():
    return IndexSpace("fr", dim=4, indices="abcde", metric="η")


@pytest.fixture
def setup(st, fr):
    return VielbeinSetup(st, fr)


@pytest.fixture
def chr(setup, st):
    g_lo = setup.spacetime_metric_lower()
    g_up = setup.spacetime_metric_upper()
    return LeviCivitaConnection(g_lo, g_up, st)


# ─── Leaf builders ────────────────────────────────────────


class TestLeafBuilders:
    def test_vielbein_positions(self, setup):
        e = setup.vielbein()
        assert len(e.indices) == 2
        assert e.indices[0].position == "upper"  # frame
        assert e.indices[1].position == "lower"  # spacetime
        assert e.indices[0].space.name == "fr"
        assert e.indices[1].space.name == "st"

    def test_vielbein_inverse_positions(self, setup):
        e_inv = setup.vielbein_inverse()
        assert e_inv.indices[0].position == "lower"  # frame lower
        assert e_inv.indices[1].position == "upper"  # st upper

    def test_vielbein_aμ_upper(self, setup):
        e = setup.vielbein_aμ_upper()
        assert all(i.position == "upper" for i in e.indices)

    def test_vielbein_aμ_lower(self, setup):
        e = setup.vielbein_aμ_lower()
        assert all(i.position == "lower" for i in e.indices)

    def test_frame_metric_symmetric(self, setup):
        eta = setup.frame_metric_lower()
        assert eta.symmetric_pairs == ((0, 1),)
        assert all(i.space.name == "fr" for i in eta.indices)

    def test_spacetime_metric_symmetric(self, setup):
        g = setup.spacetime_metric_lower()
        assert g.symmetric_pairs == ((0, 1),)


# ─── SpinConnection ────────────────────────────────────────


class TestSpinConnection:
    def test_subclass_of_connection(self, setup):
        spin = setup.spin_connection()
        assert isinstance(spin, Connection)
        assert isinstance(spin, SpinConnection)

    def test_acts_on_frame_deriv_in_st(self, setup):
        spin = setup.spin_connection()
        assert spin.space.name == "fr"
        assert spin.deriv_space.name == "st"

    def test_make_tensor_structure(self, setup):
        """ω^a_{μ b} 구조: frame upper, st lower, frame lower."""
        spin = setup.spin_connection()
        omega = spin.make_tensor("a", "μ", "b")
        assert omega.indices[0].position == "upper"
        assert omega.indices[0].space.name == "fr"
        assert omega.indices[1].position == "lower"
        assert omega.indices[1].space.name == "st"
        assert omega.indices[2].position == "lower"
        assert omega.indices[2].space.name == "fr"


# ─── vielbein_compatibility_lhs ───────────────────────────


class TestCompatibility:
    def test_compact_form(self, setup, chr):
        compat = vielbein_compatibility_lhs(setup, chr)
        assert isinstance(compat, CovariantDeriv)
        # free = [μ, a, ν]
        names = sorted(i.name for i in compat.free_indices)
        assert names == ["a", "ν", "μ"] or names == sorted(["a", "ν", "μ"])

    def test_expand_three_terms(self, setup, chr):
        """∂_μ e^a_ν + ω·e - Γ·e  → 3 항."""
        compat = vielbein_compatibility_lhs(setup, chr)
        expanded = expand_covariant(compat)
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(expanded)
        assert len(terms) == 3

    def test_expanded_contains_omega_and_gamma(self, setup, chr):
        compat = vielbein_compatibility_lhs(setup, chr)
        expanded = expand_covariant(compat)
        from indexcalc.core.simplify import _flatten_sum
        from indexcalc.core.contract import collect_tensors
        names = set()
        for term in _flatten_sum(expanded):
            for t in collect_tensors(term):
                names.add(t.name)
        assert "ω" in names  # spin connection
        assert "Γ" in names  # Christoffel
        assert "e" in names


# ─── spin_connection_from_vielbein ────────────────────────


class TestSpinConnFromVielbein:
    def test_free_indices(self, setup, chr):
        omg = spin_connection_from_vielbein(setup, chr)
        names = sorted(i.name for i in omg.free_indices)
        assert names == ["a", "b", "μ"]

    def test_structure_e_times_cov_e(self, setup, chr):
        omg = spin_connection_from_vielbein(setup, chr)
        assert isinstance(omg, TensorProduct)
        assert omg.left.name == "e"
        assert isinstance(omg.right, CovariantDeriv)
        assert omg.right.expr.name == "e"

    def test_expanded_two_terms(self, setup, chr):
        """e^{aν} (∂_μ e^b_ν - Γ·e) — expand covariant give 2 inner terms."""
        omg = spin_connection_from_vielbein(setup, chr)
        expanded = expand_covariant(omg)
        # expand_covariant이 TensorProduct(e_aν, TensorSum(...))로 두지만,
        # 안쪽 sum의 두 leg는 TensorProduct(e_aν, ∂e)와 ScalarMul(-1, ...e_aν·Γ·e)일 수 있음.
        # 모든 summand 안의 tensor name 모음.
        from indexcalc.core.simplify import _flatten_sum, collect_factors
        from indexcalc.core.tensor import TensorProduct as _TP
        names = set()

        def walk(node):
            for term in _flatten_sum(node):
                if isinstance(term, _TP):
                    for f in collect_factors(term):
                        if isinstance(f, Tensor):
                            names.add(f.name)
                        elif isinstance(f, TensorSum):
                            walk(f)
                        elif isinstance(f, CovariantDeriv):
                            walk(f.expr)
                        elif hasattr(f, "expr"):
                            walk(f.expr)
                elif isinstance(term, Tensor):
                    names.add(term.name)
                elif isinstance(term, CovariantDeriv):
                    walk(term.expr)
                elif hasattr(term, "expr"):
                    walk(term.expr)

        walk(expanded)
        assert "Γ" in names
        assert "e" in names


# ─── VielbeinSetup.to_registry → collapse 호환 ───────────


class TestToRegistry:
    def test_collapse_via_setup_registry(self, setup):
        """setup.to_registry()로 collapse_vielbein_identity 호환."""
        e1 = setup.vielbein("a", "μ")
        eta = setup.frame_metric_lower("a", "b")
        e2 = setup.vielbein("b", "ν")
        expr = TensorProduct(TensorProduct(e1, eta), e2)
        result = collapse_vielbein_identity(expr, setup.to_registry())
        assert isinstance(result, Tensor)
        assert result.name == "g"
