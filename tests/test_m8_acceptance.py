"""LIONS M8 acceptance — chiral Yukawa $\\bar L^i H^j \\epsilon_{ij} e_R$ 의
SU(2)_L × U(1)_Y × Lorentz invariance.

핵심 신규 도구 (이전 마일스톤 위에 얹음):

- **Chiral spinor rep tags** ``L_spinor`` / ``R_spinor`` / ``conj_L_spinor`` /
  ``conj_R_spinor``. ``make_lorentz_spinor_generator`` 가 group 에 등록되어
  있을 때만 자동으로 spinor action 을 재사용해 등록. 표준 SO(1,3) 변환에서는
  $\\Sigma^{ab}$ 가 chirality 와 commute (Σ ~ [γ^a, γ^b], $\\{\\gamma^a, \\gamma_5\\}=0$
  은 짝수번 등장)하므로 LH/RH 모두 동일한 spinor_action 으로 충분. 별도
  chirality identity simplifier 는 M9 candidate.

- **ε_{ij} 명시 contraction** (M3.5 우회 해소). $\\epsilon$ 은 ``reps={}`` 의
  invariant tensor — generator action 시 ZeroTensor 처리. 두 SU(N) Leibniz
  항이 ε antisymmetry + dummy renaming + canonical_form_modulo_dummies 로
  같은 body 로 normalize 될 수 있는지가 acceptance 의 비명시 조건.

검증 라그랑지안:

.. math::

    \\mathcal L_{\\rm chir} = \\bar L^i H^j \\epsilon_{ij} e_R

설정:
- $L^i_\\alpha$ : SU(2) fund upper, $Y=-\\tfrac12$, **L_spinor**.
- $\\bar L^i{}^\\alpha$ : SU(2) fund upper, $Y=+\\tfrac12$, **conj_L_spinor**
  (Hermitian conjugate 후 위치 reshuffle; SU(2) pseudoreal 활용).
- $H^j$ : SU(2) fund upper, $Y=+\\tfrac12$, Lorentz singlet.
- $\\epsilon_{ij}$ : SU(2) invariant, ``reps={}``, antisymmetric.
- $e_R^\\alpha$ : SU(2) singlet, $Y=-1$, **R_spinor**.

Contractions: $\\bar L^i \\cdot \\epsilon_{ij}$ (SU(2) i), $H^j \\cdot \\epsilon_{ij}$
(SU(2) j), $\\bar L_\\alpha \\cdot e_R^\\alpha$ (Lorentz spinor α).
"""

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum, ScalarMul
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.group import Group
from indexcalc.core.generator import (
    make_u1_generator, make_su_n_generator, make_lorentz_spinor_generator,
)
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify


# ─── Setup: chiral SM-lite ─────────────────────────────────


@pytest.fixture
def setup():
    """SU(2)_L × U(1)_Y × Lorentz (chiral reps) + index spaces."""
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    su2_adj = IndexSpace("su2_adj", dim=3, indices="ABCDE", metric="δ")
    su2_fund = IndexSpace("su2_fund", dim=2, indices="ijklmn")
    dirac = IndexSpace("dirac", dim=4, indices="αβγδε")

    su2 = Group("SU(2)", dim=3, abelian=False)
    su2.add_rep("adj", dim=3)
    su2.add_rep("fund", dim=2)
    su2.add_rep("antifund", dim=2, conjugate=True)
    su2.add_rep("singlet", dim=1)

    u1y = Group("U(1)_Y", dim=1, abelian=True)
    u1y.add_rep("+1/2", dim=1, charge=0.5)
    u1y.add_rep("-1/2", dim=1, charge=-0.5)
    u1y.add_rep("+1", dim=1, charge=1.0)
    u1y.add_rep("-1", dim=1, charge=-1.0)
    u1y.add_rep("0", dim=1, charge=0.0)

    lorentz = Group("Lorentz", dim=6, abelian=False)
    # chiral reps
    lorentz.add_rep("L_spinor", dim=2)
    lorentz.add_rep("R_spinor", dim=2)
    lorentz.add_rep("conj_L_spinor", dim=2, conjugate=True)
    lorentz.add_rep("conj_R_spinor", dim=2, conjugate=True)
    # legacy Dirac reps (필요시 사용)
    lorentz.add_rep("spinor", dim=4)
    lorentz.add_rep("conj_spinor", dim=4, conjugate=True)
    lorentz.add_rep("vector", dim=4)
    lorentz.add_rep("singlet", dim=1)

    su2_gen = make_su_n_generator(
        su2, su2_adj, parameter_name="P", fund_space=su2_fund,
    )
    u1y_gen = make_u1_generator(u1y, name="T_U(1)_Y")
    lorentz_gen = make_lorentz_spinor_generator(
        lorentz, frame_space=st, spinor_space=dirac,
    )

    return {
        "st": st, "su2_adj": su2_adj, "su2_fund": su2_fund, "dirac": dirac,
        "su2": su2, "u1y": u1y, "lorentz": lorentz,
        "su2_gen": su2_gen, "u1y_gen": u1y_gen, "lorentz_gen": lorentz_gen,
    }


# ─── Field constructors ────────────────────────────────────


def make_L(s, i_name="i", α_name="α"):
    """$L^i_\\alpha$ — SU(2) fund upper, Y=-1/2, L_spinor."""
    return Tensor(
        "L",
        [s["su2_fund"].upper(i_name), s["dirac"].upper(α_name)],
        reps={"SU(2)": "fund", "U(1)_Y": "-1/2", "Lorentz": "L_spinor"},
        statistics="fermionic",
    )


def make_Lbar(s, i_name="i", α_name="α"):
    """$\\bar L^i{}_\\alpha$ — SU(2) fund upper (pseudoreal), Y=+1/2, conj_L_spinor."""
    return Tensor(
        "Lbar",
        [s["su2_fund"].upper(i_name), s["dirac"].lower(α_name)],
        reps={"SU(2)": "fund", "U(1)_Y": "+1/2", "Lorentz": "conj_L_spinor"},
        statistics="fermionic",
    )


def make_H(s, j_name="j"):
    return Tensor(
        "H",
        [s["su2_fund"].upper(j_name)],
        reps={"SU(2)": "fund", "U(1)_Y": "+1/2", "Lorentz": "singlet"},
    )


def make_eR(s, α_name="α"):
    """$e_R^\\alpha$ — SU(2) singlet, Y=-1, R_spinor."""
    return Tensor(
        "eR",
        [s["dirac"].upper(α_name)],
        reps={"SU(2)": "singlet", "U(1)_Y": "-1", "Lorentz": "R_spinor"},
        statistics="fermionic",
    )


def make_epsilon_lower(s, i_name="i", j_name="j"):
    """$\\epsilon_{ij}$ — SU(2) invariant, antisymmetric, ``reps={}`` 로 invariant 처리."""
    return Tensor(
        "epsilon",
        [s["su2_fund"].lower(i_name), s["su2_fund"].lower(j_name)],
        antisymmetric_pairs=[(0, 1)],
    )


# ─── M8-Lorentz: chiral spinor rep contraction ──────────────


def test_chiral_yukawa_lorentz_invariant(setup):
    """$\\delta_{\\rm Lorentz}(\\bar L^i H^j \\epsilon_{ij} e_R) = 0$.

    M4 mass-term path 와 동일 — conj_L_spinor 의 +i/2 ψ̄Σ 항과 R_spinor 의
    -i/2 Σψ 항이 직접 spinor contraction (α dummy) 으로 cancel.
    """
    s = setup
    Lbar = make_Lbar(s, "i", "α")
    H = make_H(s, "j")
    eps = make_epsilon_lower(s, "i", "j")
    eR = make_eR(s, "α")

    L_yuk = TensorProduct(Lbar, TensorProduct(H, TensorProduct(eps, eR)))

    delta = apply_generator(L_yuk, s["lorentz_gen"])
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {type(final).__name__}: {final!r}"


def test_chiral_yukawa_u1y_invariant(setup):
    """$Y_{\\bar L} + Y_H + Y_\\epsilon + Y_{e_R} = +\\tfrac12 + \\tfrac12 + 0 + (-1) = 0$.

    ε 은 hypercharge 0, U(1) phase 합산 → 0.
    """
    s = setup
    Lbar = make_Lbar(s, "i", "α")
    H = make_H(s, "j")
    eps = make_epsilon_lower(s, "i", "j")
    eR = make_eR(s, "α")

    L_yuk = TensorProduct(Lbar, TensorProduct(H, TensorProduct(eps, eR)))

    delta = apply_generator(L_yuk, s["u1y_gen"])
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {type(final).__name__}: {final!r}"


def test_chiral_yukawa_su2_invariant(setup):
    """$\\delta_{SU(2)}(\\bar L^i H^j \\epsilon_{ij} e_R) = 0$.

    핵심 단계 — ε 명시 contraction 의 SU(2) invariance (M3.5 에서 우회한 부분).
    SU(2) Leibniz 두 항:
        (a)  i (T^P)^i{}_k \\bar L^k H^j \\epsilon_{ij} e_R
        (b)  i \\bar L^i (T^P)^j{}_k H^k \\epsilon_{ij} e_R
    ε invariance ((T^P)^p_q ε_{pr} + (T^P)^p_r ε_{qp} = 0) + ε antisymmetric
    + dummy renaming 으로 (a) + (b) = 0.
    """
    s = setup
    Lbar = make_Lbar(s, "i", "α")
    H = make_H(s, "j")
    eps = make_epsilon_lower(s, "i", "j")
    eR = make_eR(s, "α")

    L_yuk = TensorProduct(Lbar, TensorProduct(H, TensorProduct(eps, eR)))

    delta = apply_generator(L_yuk, s["su2_gen"])
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {type(final).__name__}: {final!r}"


# ─── M8 negative / sanity ──────────────────────────────────


def test_chiral_yukawa_no_epsilon_not_invariant(setup):
    """ε 없이 $\\bar L^i H^j e_R$ — SU(2) doublet × doublet (no antisym contraction),
    SU(2) invariance 통과하면 false positive (검증기 오류). ZeroTensor 아니어야 함.
    """
    s = setup
    Lbar = make_Lbar(s, "i", "α")
    H = make_H(s, "j")
    eR = make_eR(s, "α")

    L_bad = TensorProduct(Lbar, TensorProduct(H, eR))
    delta = apply_generator(L_bad, s["su2_gen"])
    final = simplify(delta)
    assert not isinstance(final, ZeroTensor), (
        f"false positive: ε 없는 doublet×doublet 이 SU(2) invariant 로 잘못 판정됨"
    )


def test_chiral_yukawa_full_invariance(setup):
    """세 generator (SU(2), U(1)_Y, Lorentz) 모두에 대해 한 라그랑지안 동시 invariant."""
    s = setup
    Lbar = make_Lbar(s, "i", "α")
    H = make_H(s, "j")
    eps = make_epsilon_lower(s, "i", "j")
    eR = make_eR(s, "α")
    L_yuk = TensorProduct(Lbar, TensorProduct(H, TensorProduct(eps, eR)))

    for gen_key in ["su2_gen", "u1y_gen", "lorentz_gen"]:
        delta = apply_generator(L_yuk, s[gen_key])
        final = simplify(delta)
        assert isinstance(final, ZeroTensor), (
            f"FAILED for {gen_key}: got {type(final).__name__}: {final!r}"
        )
