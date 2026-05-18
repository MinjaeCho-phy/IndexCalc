"""D22: U(N)/SU(N) probe via classical_group_spec.

검증 toy:
1. SU(2) fund bilinear $\\bar\\phi_I \\phi^I$ — SU(2) invariant.
2. SU(2) fund × wrong rep — invariance 깨짐 catch.
3. U(1) charge ±1 bilinear $\\bar\\phi \\phi$ — U(1) invariant.
4. U(1) charged single field — non-invariant.
5. classical_group_spec("SU(2)", 2, fund_space) 기본 동작.
"""

from __future__ import annotations

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor
from indexcalc.lions.probe import (
    GroupSpec, ProbeResult, probe, classical_group_spec,
)


@pytest.fixture
def su2_fund():
    return IndexSpace("su2_fund", dim=2, indices="IJKLMN")


@pytest.fixture
def su2_spec(su2_fund):
    return classical_group_spec("SU(2)", 2, su2_fund)


@pytest.fixture
def u1_spec():
    return classical_group_spec("U(1)", 1)


def test_classical_group_spec_su2_smoke(su2_fund, su2_spec):
    assert su2_spec.name == "SU(2)"
    assert su2_spec.dim == 3  # SU(2): dim = 4 - 1 = 3
    assert su2_spec.group.has_rep("fund")
    assert su2_spec.group.has_rep("antifund")
    assert su2_spec.group.has_rep("singlet")
    assert su2_spec.generator.has_action("fund")
    assert su2_spec.generator.has_action("antifund")


def test_su2_fund_bilinear_invariant(su2_fund, su2_spec):
    """$\\bar\\phi_I \\phi^I$ — SU(2) fund × antifund 결합, invariant."""
    phi = Tensor("phi", [su2_fund.upper("I")],
                 reps={"SU(2)": "fund"})
    phidag = Tensor("phidag", [su2_fund.lower("I")],
                    reps={"SU(2)": "antifund"})
    L = phidag * phi
    r = probe(L, [phi, phidag], [su2_spec])[0]
    assert r.invariant is True, (
        f"SU(2) fund-bilinear invariance fail: notes={r.notes}"
    )
    assert r.non_singlet_fields == {
        "phi": "fund", "phidag": "antifund",
    }
    assert r.dim == 3


def test_u1_neutral_bilinear_invariant(u1_spec):
    """$\\bar\\phi \\phi$ with charges (+1, -1) — net charge 0, invariant."""
    phi_plus = Tensor("phi_p", [], reps={"U(1)": "+1"})
    phi_minus = Tensor("phi_m", [], reps={"U(1)": "-1"})
    L = phi_minus * phi_plus
    r = probe(L, [phi_plus, phi_minus], [u1_spec])[0]
    assert r.invariant is True
    assert r.dim == 1


def test_u1_charged_single_field_not_invariant(u1_spec):
    """단일 charged field — net charge ≠ 0이라 U(1) 변환에 invariant 아님."""
    phi = Tensor("phi", [], reps={"U(1)": "+1"})
    r = probe(phi, [phi], [u1_spec])[0]
    assert r.invariant is False


def test_su2_with_explicit_singlet_passes(su2_fund, su2_spec):
    """fund × antifund contracted + 외부 singlet field 곱 — invariant."""
    phi = Tensor("phi", [su2_fund.upper("I")], reps={"SU(2)": "fund"})
    phidag = Tensor("phidag", [su2_fund.lower("I")], reps={"SU(2)": "antifund"})
    sing = Tensor("c", [], reps={"SU(2)": "singlet"})
    L = sing * phidag * phi
    r = probe(L, [phi, phidag, sing], [su2_spec])[0]
    assert r.invariant is True


def test_u_n_with_explicit_adj_space():
    """U(2) GroupSpec — dim = 4, adj_space 명시 가능."""
    fund = IndexSpace("u2_fund", dim=2, indices="IJK")
    adj = IndexSpace("u2_adj", dim=4, indices="abcd")
    u2 = classical_group_spec("U(2)", 2, fund, adj_space=adj)
    assert u2.dim == 4
    assert u2.generator.has_action("fund")
