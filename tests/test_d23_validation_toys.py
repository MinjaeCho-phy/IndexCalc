"""D23: validation toy 확장 — probe correctness across diverse Lagrangians.

검증 toy:
1. Multi-field SO(3) bilinear $A^i B_i$ — invariant, 두 필드 모두 vector.
2. SO(3) ε-trilinear $\\epsilon_{ijk} A^i B^j C^k$ — SO(3) ✓ O(3) ✗ (parity).
3. Two-vector-fields with cross-rotation under same group.
4. Mixed-group Lagrangian: SO(3) × SU(2). 같은 필드가 두 그룹의 rep을 가짐.
5. Anisotropic break via wrong-rep field (component selection 대용).
"""

from __future__ import annotations

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor
from indexcalc.lions.probe import (
    probe, classical_group_spec, format_probe_report,
)


@pytest.fixture
def so3_vec():
    return IndexSpace("so3_vec", dim=3, indices="ijklmn", metric="delta")


@pytest.fixture
def so3_spec(so3_vec):
    return classical_group_spec("SO(3)", 3, so3_vec)


@pytest.fixture
def o3_spec(so3_vec):
    return classical_group_spec("O(3)", 3, so3_vec)


@pytest.fixture
def su2_fund():
    return IndexSpace("su2_fund", dim=2, indices="IJKLMN")


@pytest.fixture
def su2_spec(su2_fund):
    return classical_group_spec("SU(2)", 2, su2_fund)


def test_two_vector_fields_bilinear_so3(so3_vec, so3_spec):
    """$A^i B_i$ — 두 vector 필드 결합. probe는 둘 다 non_singlet에 포함."""
    A = Tensor("A", [so3_vec.upper("i")], reps={"SO(3)": "vector"})
    B = Tensor("B", [so3_vec.lower("i")], reps={"SO(3)": "vector"})
    L = A * B
    r = probe(L, [A, B], [so3_spec])[0]
    assert r.invariant is True
    assert r.non_singlet_fields == {"A": "vector", "B": "vector"}
    assert r.dim == 3


def test_epsilon_trilinear_so3_yes_o3_no(so3_vec, so3_spec, o3_spec):
    """$\\epsilon_{ijk} A^i B^j C^k$ — SO(3) ✓, O(3) ✗ (parity flip)."""
    A = Tensor("A", [so3_vec.upper("i")], reps={"SO(3)": "vector", "O(3)": "vector"})
    B = Tensor("B", [so3_vec.upper("j")], reps={"SO(3)": "vector", "O(3)": "vector"})
    C = Tensor("C", [so3_vec.upper("k")], reps={"SO(3)": "vector", "O(3)": "vector"})
    eps = Tensor(
        "epsilon",
        [so3_vec.lower("i"), so3_vec.lower("j"), so3_vec.lower("k")],
        antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
    )
    L = eps * A * B * C

    rs = {r.group: r for r in probe(L, [A, B, C], [so3_spec, o3_spec])}
    assert rs["SO(3)"].invariant is True, (
        f"SO(3) ε-trilinear should be invariant: {rs['SO(3)']}"
    )
    assert rs["O(3)"].invariant is False, (
        f"O(3) ε-trilinear should NOT be invariant (parity): {rs['O(3)']}"
    )


def test_mixed_group_fields_so3_and_su2(so3_vec, su2_fund, so3_spec, su2_spec):
    """L = $\\delta_{ij} A^i B^j \\bar\\phi_I \\phi^I$ — SO(3) × SU(2) 둘 다 invariant.

    두 그룹의 rep을 동시에 가진 필드는 없지만 그룹간 인덱스가 독립 contract.
    """
    A = Tensor("A", [so3_vec.upper("i")], reps={"SO(3)": "vector"})
    B = Tensor("B", [so3_vec.upper("j")], reps={"SO(3)": "vector"})
    delta = Tensor("delta", [so3_vec.lower("i"), so3_vec.lower("j")],
                   symmetric_pairs=[(0, 1)], reps={"SO(3)": "singlet"})
    phi = Tensor("phi", [su2_fund.upper("I")], reps={"SU(2)": "fund"})
    phidag = Tensor("phidag", [su2_fund.lower("I")],
                    reps={"SU(2)": "antifund"})
    L = delta * A * B * phidag * phi

    results = probe(L, [A, B, phi, phidag], [so3_spec, su2_spec])
    by_group = {r.group: r for r in results}
    assert by_group["SO(3)"].invariant is True
    assert by_group["SU(2)"].invariant is True
    assert by_group["SO(3)"].non_singlet_fields == {"A": "vector", "B": "vector"}
    assert by_group["SU(2)"].non_singlet_fields == {
        "phi": "fund", "phidag": "antifund"
    }


def test_anisotropic_break_wrong_rep(so3_vec, so3_spec):
    """Anisotropic symmetry-break: external vector source $J^i$ singlet-tagged.

    Component selection 표현이 abstract IR에 없어서 이렇게 대용: 벡터 인덱스를
    가진 필드를 singlet으로 잘못 tagging — probe는 rep mismatch로 invariant=False.
    """
    Phi = Tensor("Phi", [so3_vec.upper("i")], reps={"SO(3)": "vector"})
    J = Tensor("J", [so3_vec.lower("i")], reps={"SO(3)": "singlet"})  # 잘못 tag
    L = Phi * J
    r = probe(L, [Phi, J], [so3_spec])[0]
    assert r.invariant is False


def test_report_format_multi_group(so3_vec, su2_fund, so3_spec, su2_spec):
    """Mixed-group report 형식 sanity."""
    A = Tensor("A", [so3_vec.upper("i")], reps={"SO(3)": "vector"})
    B = Tensor("B", [so3_vec.lower("i")], reps={"SO(3)": "vector"})
    L = A * B
    results = probe(L, [A, B], [so3_spec, su2_spec])
    text = format_probe_report("Two-vector bilinear", results)
    assert "SO(3)" in text and "✓" in text
    # SU(2)는 fields가 singlet (rep 미등록) — invariant but trivial.
    assert "SU(2)" in text


def test_two_independent_so3_vectors_inner_product_invariant(so3_vec, so3_spec):
    """$A^i B_i$ + $\\delta_{ij} C^i C^j$ — 두 invariant 항의 합."""
    A = Tensor("A", [so3_vec.upper("i")], reps={"SO(3)": "vector"})
    B = Tensor("B", [so3_vec.lower("i")], reps={"SO(3)": "vector"})
    C1 = Tensor("C", [so3_vec.upper("j")], reps={"SO(3)": "vector"})
    C2 = Tensor("C", [so3_vec.upper("k")], reps={"SO(3)": "vector"})
    delta = Tensor("delta", [so3_vec.lower("j"), so3_vec.lower("k")],
                   symmetric_pairs=[(0, 1)], reps={"SO(3)": "singlet"})
    L = (A * B) + (delta * C1 * C2)
    r = probe(L, [A, B, C1], [so3_spec])[0]
    assert r.invariant is True
