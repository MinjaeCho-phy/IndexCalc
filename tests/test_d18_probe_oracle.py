"""D18: probe() oracle on NR mechanics Lagrangians.

검증 toy:
1. Kepler L = (1/2) δ_ij \\dot Φ^i \\dot Φ^j + κ/sqrt(δ_kl Φ^k Φ^l)
   → SO(3) ✓ O(3) ✓ (vector field Φ는 non-trivial)
2. Free Schrödinger-like real scalar (no SO(3) action): non-singlet 없음 → trivial
3. Harmonic oscillator: (1/2) \\dot Φ^2 - (1/2) ω² Φ^2 → SO(N) ✓
4. Hard negative: \\dot Φ_i Φ_i + Φ_1² (rotation 깨짐) — 1번 컴포넌트 단독 텀
   → SO(3)/O(3) ✗

NOTE: Hard-negative toy 4는 사용자 spec의 "intentional break" — IR로
표현하려면 component-level (Φ_1만 선택) 텐서가 필요. v2 IR은 abstract
index만 다루므로 toy 4는 일단 "Φ × Y (Y는 singlet 외부 source)" 같은
mismatched-index 형태로 대체.
"""

from __future__ import annotations

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum, ScalarMul
from indexcalc.core.scalar_function import ScalarFunction
from indexcalc.adm import TimeDeriv
from indexcalc.lions.probe import (
    GroupSpec, ProbeResult, probe, classical_group_spec, format_probe_report,
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


def _build_kepler(so3_vec):
    """Kepler L IR 빌드 (rep="vector" tagged)."""
    # 운동항
    Phi_i = Tensor("Phi", [so3_vec.upper("i")],
                   reps={"SO(3)": "vector", "O(3)": "vector"})
    Phi_j = Tensor("Phi", [so3_vec.upper("j")],
                   reps={"SO(3)": "vector", "O(3)": "vector"})
    delta_ij = Tensor("delta", [so3_vec.lower("i"), so3_vec.lower("j")],
                      symmetric_pairs=[(0, 1)],
                      reps={"SO(3)": "singlet", "O(3)": "singlet"})
    kinetic = ScalarMul(
        0.5, delta_ij * TimeDeriv(Phi_i) * TimeDeriv(Phi_j),
    )
    # 퍼텐셜
    Phi_k = Tensor("Phi", [so3_vec.upper("k")],
                   reps={"SO(3)": "vector", "O(3)": "vector"})
    Phi_l = Tensor("Phi", [so3_vec.upper("l")],
                   reps={"SO(3)": "vector", "O(3)": "vector"})
    delta_kl = Tensor("delta", [so3_vec.lower("k"), so3_vec.lower("l")],
                      symmetric_pairs=[(0, 1)],
                      reps={"SO(3)": "singlet", "O(3)": "singlet"})
    r_sq = delta_kl * Phi_k * Phi_l
    potential = ScalarFunction("inv_sqrt", r_sq)
    return TensorSum(kinetic, potential), [Phi_i]  # fields for non-singlet dump


def test_kepler_is_so3_invariant(so3_vec, so3_spec):
    L, fields = _build_kepler(so3_vec)
    results = probe(L, fields, [so3_spec])
    assert len(results) == 1
    r = results[0]
    assert r.group == "SO(3)"
    assert r.invariant is True
    assert r.non_singlet_fields == {"Phi": "vector"}
    assert r.dim == 3


def test_kepler_is_o3_invariant(so3_vec, o3_spec):
    """O(3)도 보존 — algebra 수준에서 SO(3)와 동일 (improper rotation은 별도)."""
    L, fields = _build_kepler(so3_vec)
    results = probe(L, fields, [o3_spec])
    assert results[0].invariant is True
    assert results[0].dim == 3


def test_harmonic_oscillator_so3_invariant(so3_vec, so3_spec):
    """(1/2) \\dot Φ_i \\dot Φ^i - (1/2) ω² Φ_i Φ^i."""
    Phi_up = Tensor("Phi", [so3_vec.upper("i")],
                    reps={"SO(3)": "vector"})
    Phi_dn = Tensor("Phi", [so3_vec.lower("i")],
                    reps={"SO(3)": "vector"})
    kinetic = ScalarMul(0.5, TimeDeriv(Phi_up) * TimeDeriv(Phi_dn))
    Phi_up2 = Tensor("Phi", [so3_vec.upper("j")],
                     reps={"SO(3)": "vector"})
    Phi_dn2 = Tensor("Phi", [so3_vec.lower("j")],
                     reps={"SO(3)": "vector"})
    mass_term = ScalarMul(-0.5, Phi_up2 * Phi_dn2)
    L = TensorSum(kinetic, mass_term)
    r = probe(L, [Phi_up], [so3_spec])[0]
    assert r.invariant is True
    assert r.non_singlet_fields == {"Phi": "vector"}


def test_scalar_only_is_trivial(so3_vec, so3_spec):
    """모든 field가 singlet — SO(3) trivially invariant, notes에 표시."""
    phi = Tensor("phi", [], reps={"SO(3)": "singlet"})
    phi2 = Tensor("phi", [], reps={"SO(3)": "singlet"})
    L = phi * phi2  # rank-0 product
    r = probe(L, [phi], [so3_spec])[0]
    assert r.invariant is True
    assert r.non_singlet_fields == {}
    assert "trivial" in r.notes


def test_rotation_breaking_term_fails_probe(so3_vec, so3_spec):
    """깨진 회전 대칭 hard-negative: A^i B_i + C^i C_i 처럼 보이지만 C는 다른
    rep로 잘못 tagged인 경우 — generator action 결과가 non-zero."""
    # Φ^i (vector) 와 ψ^i (잘못 표기된 singlet — 외부에서 강제로 vector index)
    # 이 조합은 ψ가 singlet으로 등록되어 있어 회전에 안 변하지만 index 구조는
    # 회전 매트릭스가 작용하길 요구 → mismatch → δL ≠ 0.
    # 실제 oracle은 ψ에 대해 singlet action (=0)을 적용, Φ에 대해 vector
    # action (=M·Φ)을 적용. 결과: M^{ab} Φ^... ψ^...의 contracted term이
    # 남아 ZeroTensor로 simplify 안 됨.
    Phi = Tensor("Phi", [so3_vec.upper("i")],
                 reps={"SO(3)": "vector"})
    psi = Tensor("psi", [so3_vec.lower("i")],
                 reps={"SO(3)": "singlet"})  # 의도적으로 singlet
    L = Phi * psi
    r = probe(L, [Phi, psi], [so3_spec])[0]
    assert r.invariant is False, (
        f"rep 미스매치(vector × singlet)인데 invariant로 판정됨: "
        f"non_singlet_fields={r.non_singlet_fields}"
    )


def test_format_report_smoke(so3_vec, so3_spec, o3_spec):
    L, fields = _build_kepler(so3_vec)
    results = probe(L, fields, [so3_spec, o3_spec])
    text = format_probe_report("Kepler L", results)
    assert "SO(3)" in text
    assert "O(3)" in text
    assert "vector" in text
    assert "dim=3" in text
