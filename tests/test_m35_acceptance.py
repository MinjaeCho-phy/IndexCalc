"""LIONS M3.5 acceptance — Higgs sector $|H|^2$ invariance under SU(2) × U(1)_Y.

검증 대상: $\\bar H H = \\bar H_i H^i$ scalar, gauge group $SU(2)_L \\times U(1)_Y$의
**global** 변환에 대해 invariant.

핵심 도구:
- ``su_n_fund_action`` (M3.5-B): fund / antifund rep 작용. $\\delta_a H^i = i T^{a,i}{}_j H^j$.
- ``canonical_form_modulo_dummies`` (M3.5-A): dummy 인덱스 relabeling을 인식해
  Leibniz 두 항이 dummy 이름만 다른 경우 같은 group으로 묶음.

**제외 (M4 예정):**
- Local gauge invariance ($\\partial_\\mu \\alpha$ 매개변수, $D_\\mu H$).
- Higgs potential의 explicit form ($-\\mu^2 |H|^2 + \\lambda |H|^4$ — quartic).
- Yukawa 결합 ($\\bar Q_L \\tilde H u_R$ — chirality + epsilon contractions).
- Lorentz invariance.
"""

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.group import Group
from indexcalc.core.generator import (
    make_u1_generator, make_su_n_generator,
)
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify


# ─── Setup ──────────────────────────────────────────────────


@pytest.fixture
def setup():
    """SU(2)_L × U(1)_Y + Higgs doublet H, antidoublet barH."""
    # SU(2)_L
    su2 = Group("SU(2)", dim=3, abelian=False)
    su2.add_rep("fund", dim=2)
    su2.add_rep("antifund", dim=2, conjugate=True)
    su2.add_rep("adj", dim=3)
    su2.add_rep("singlet", dim=1)

    # U(1)_Y
    u1y = Group("U(1)_Y", dim=1, abelian=True)
    u1y.add_rep("+1/2", dim=1, charge=0.5)
    u1y.add_rep("-1/2", dim=1, charge=-0.5)
    u1y.add_rep("0", dim=1, charge=0.0)

    # Index spaces
    su2_adj = IndexSpace("su2_adj", dim=3, indices="abc", metric="δ")
    su2_fund = IndexSpace("su2_fund", dim=2, indices="ij")
    # fund 자체엔 metric 없음 (raise/lower는 epsilon으로 — M4)

    su2_gen = make_su_n_generator(
        su2, su2_adj,
        parameter_name="A", fund_space=su2_fund,
    )
    u1y_gen = make_u1_generator(u1y, name="T_U(1)_Y")

    return su2, u1y, su2_adj, su2_fund, su2_gen, u1y_gen


def make_H(su2_fund, idx_name="i"):
    """$H^i$ — SU(2) doublet, U(1)_Y charge +1/2."""
    return Tensor(
        "H",
        [su2_fund.upper(idx_name)],
        reps={"SU(2)": "fund", "U(1)_Y": "+1/2"},
    )


def make_Hbar(su2_fund, idx_name="i"):
    """$\\bar H_i$ — SU(2) antidoublet, U(1)_Y charge -1/2."""
    return Tensor(
        "Hbar",
        [su2_fund.lower(idx_name)],
        reps={"SU(2)": "antifund", "U(1)_Y": "-1/2"},
    )


# ─── 1. Higgs scalar bilinear invariance ────────────────────


def test_Hbar_H_scalar_su2_invariant(setup):
    """$\\delta_{\\rm SU(2)}(\\bar H_i H^i) = 0$.

    Leibniz의 두 항은 ``T^{a,i}_j``의 i, j가 swap된 형태로 dummy renaming 없이는
    별개 expression이지만, ``canonical_form_modulo_dummies``로 같은 body로 인식
    → ``collect_scalar_terms``가 +i와 -i 합산해 0 검출.
    """
    *_, su2_fund, su2_gen, _ = setup
    H = make_H(su2_fund, "i")
    Hbar = make_Hbar(su2_fund, "i")
    L = TensorProduct(Hbar, H)  # \bar H_i H^i

    delta = apply_generator(L, su2_gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


def test_Hbar_H_scalar_u1y_invariant(setup):
    """$\\delta_{U(1)_Y}(\\bar H H) = 0$ — phase cancellation (M3와 같은 메커니즘)."""
    *_, su2_fund, _, u1y_gen = setup
    H = make_H(su2_fund, "i")
    Hbar = make_Hbar(su2_fund, "i")
    L = TensorProduct(Hbar, H)

    delta = apply_generator(L, u1y_gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor)


def test_Hbar_H_scalar_full_invariance(setup):
    """양쪽 generator 다 invariance 통과."""
    *_, su2_fund, su2_gen, u1y_gen = setup
    H = make_H(su2_fund, "i")
    Hbar = make_Hbar(su2_fund, "i")
    L = TensorProduct(Hbar, H)

    for gen in [su2_gen, u1y_gen]:
        delta = apply_generator(L, gen)
        final = simplify(delta)
        assert isinstance(final, ZeroTensor), f"failed for {gen.name!r}"


# ─── 2. Negative — H H (두 같은 rep) NOT invariant ──────────


def test_two_H_not_invariant_under_su2(setup):
    """$H^i H^j$ — SU(2) variation은 0 아님 (두 fund의 곱은 fund⊗fund 텐서)."""
    *_, su2_fund, su2_gen, _ = setup
    Hi = make_H(su2_fund, "i")
    Hj = make_H(su2_fund, "j")
    L = TensorProduct(Hi, Hj)

    delta = apply_generator(L, su2_gen)
    final = simplify(delta)
    # SU(2) variation: $\\delta(H^i H^j) = i T^{a,i}_k H^k H^j + i H^i T^{a,j}_k H^k$
    # 일반적으로 0 아님 (singlet으로 묶이지 않음).
    assert not isinstance(final, ZeroTensor)


def test_Hbar_only_not_invariant(setup):
    """$\\bar H_i$ 단독 — 단일 fund 인덱스 free, U(1) phase가 살아있음."""
    *_, su2_fund, _, u1y_gen = setup
    Hbar = make_Hbar(su2_fund, "i")
    delta = apply_generator(Hbar, u1y_gen)
    final = simplify(delta)
    # δ\bar H = -i (1/2) \bar H — 0 아님
    assert not isinstance(final, ZeroTensor)


# ─── 3. canonical_form_modulo_dummies 동작 확인 ──────────────


def test_canonical_form_dummy_relabeling(setup):
    """dummy 이름만 다른 두 expression의 canonical form이 같음."""
    from indexcalc.core.simplify import canonical_form_modulo_dummies

    *_, su2_fund, _, _ = setup
    H_i = make_H(su2_fund, "i")
    H_j = make_H(su2_fund, "j")
    Hbar_i = make_Hbar(su2_fund, "i")
    Hbar_j = make_Hbar(su2_fund, "j")

    e1 = TensorProduct(Hbar_i, H_i)  # \bar H_i H^i
    e2 = TensorProduct(Hbar_j, H_j)  # \bar H_j H^j (same expression, dummy renamed)

    assert canonical_form_modulo_dummies(e1) == canonical_form_modulo_dummies(e2)
