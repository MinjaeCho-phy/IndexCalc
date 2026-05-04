"""LIONS M5 acceptance — SM gauge sector assembly demo.

M1~M4까지의 도구로 SM-lite 라그랑지안의 **gauge invariance**를 한꺼번에 검증.
다중 그룹 ($SU(2)_L$, $U(1)_Y$, Lorentz), 멀티-rep 필드, 4-field 항까지 포함.

검증 라그랑지안 ($\\mathcal L_{\\rm SM-lite}$):
    - lepton mass:    $-m \\bar\\psi^i_\\alpha \\psi_i^\\alpha$   ($SU(2)$ 더블렛 × Dirac)
    - Higgs mass:     $-\\mu^2 \\bar H_i H^i$
    - Higgs quartic:  $\\lambda (\\bar H_i H^i)(\\bar H_j H^j)$
    - Yukawa-like:    $-y \\bar\\psi^i_\\alpha H_i \\chi^\\alpha$  ($\\chi$ = SU(2) singlet 페르미온)
    - SU(2) YM:       $-\\tfrac14 W^A_{\\mu\\nu} W_A^{\\mu\\nu}$  (M2.5 패턴)

각 generator (SU(2), U(1)_Y, Lorentz spinor)에 대해 $\\delta\\mathcal L = 0$.

**제외 (M5+ 차후):**
- 자유 Dirac 운동항 $i\\bar\\psi\\gamma^\\mu\\partial_\\mu\\psi$ Lorentz invariance — Clifford
  simplifier 필요.
- $\\tilde H = \\epsilon H^*$ 명시적 처리 — fund/antifund 직접 contraction으로 우회.
- 본격 chirality $P_{L,R}$ projectors — Yukawa는 Dirac 형태로 simplify.
- Local gauge ($\\partial_\\mu\\alpha$).
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


# ─── Setup: SM-lite 그룹/공간/필드 ──────────────────────────


@pytest.fixture
def setup_sm():
    """SU(2)_L × U(1)_Y × Lorentz + 모든 인덱스 공간."""
    # Index spaces
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    su2_adj = IndexSpace("su2_adj", dim=3, indices="ABCDE", metric="δ")
    su2_fund = IndexSpace("su2_fund", dim=2, indices="ij")
    dirac = IndexSpace("dirac", dim=4, indices="αβγδε")  # no metric

    # SU(2)_L
    su2 = Group("SU(2)", dim=3, abelian=False)
    su2.add_rep("adj", dim=3)
    su2.add_rep("fund", dim=2)
    su2.add_rep("antifund", dim=2, conjugate=True)
    su2.add_rep("singlet", dim=1)

    # U(1)_Y — charge 양·음 0.5와 0
    u1y = Group("U(1)_Y", dim=1, abelian=True)
    u1y.add_rep("+1/2", dim=1, charge=0.5)
    u1y.add_rep("-1/2", dim=1, charge=-0.5)
    u1y.add_rep("+1", dim=1, charge=1.0)
    u1y.add_rep("-1", dim=1, charge=-1.0)
    u1y.add_rep("0", dim=1, charge=0.0)

    # Lorentz
    lorentz = Group("Lorentz", dim=6, abelian=False)
    lorentz.add_rep("spinor", dim=4)
    lorentz.add_rep("conj_spinor", dim=4, conjugate=True)
    lorentz.add_rep("vector", dim=4)
    lorentz.add_rep("singlet", dim=1)

    # Generators
    # parameter_name "P" — W field의 adj 인덱스 "A"와 충돌 회피
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


def make_psi(s, i_name="i", α_name="α"):
    """$\\psi^i_\\alpha$ — SU(2) doublet × Lorentz spinor, $Y = -1/2$."""
    return Tensor(
        "psi",
        [s["su2_fund"].upper(i_name), s["dirac"].upper(α_name)],
        reps={
            "SU(2)": "fund", "U(1)_Y": "-1/2", "Lorentz": "spinor",
        },
        statistics="fermionic",
    )


def make_psibar(s, i_name="i", α_name="α"):
    """$\\bar\\psi_i^\\alpha$ — SU(2) antidoublet × Dirac conj, $Y = +1/2$.

    ``\\bar\\psi`` is the Dirac adjoint, transforms as conj_spinor in Lorentz.
    SU(2) index lowered (antifund). Charge opposite to ψ.
    """
    return Tensor(
        "psibar",
        [s["su2_fund"].lower(i_name), s["dirac"].lower(α_name)],
        reps={
            "SU(2)": "antifund", "U(1)_Y": "+1/2", "Lorentz": "conj_spinor",
        },
        statistics="fermionic",
    )


def make_H(s, i_name="i"):
    """$H^i$ — SU(2) doublet, $Y = +1/2$, Lorentz scalar."""
    return Tensor(
        "H",
        [s["su2_fund"].upper(i_name)],
        reps={"SU(2)": "fund", "U(1)_Y": "+1/2", "Lorentz": "singlet"},
    )


def make_Hbar(s, i_name="i"):
    return Tensor(
        "Hbar",
        [s["su2_fund"].lower(i_name)],
        reps={"SU(2)": "antifund", "U(1)_Y": "-1/2", "Lorentz": "singlet"},
    )


def make_chi(s, α_name="α"):
    """$\\chi^\\alpha$ — SU(2) singlet, Lorentz spinor (right-handed-like).

    Yukawa balance에 쓰일 charge는 specific 항에서 부여 (test마다 다를 수 있음).
    여기선 ``Y = -1`` (Yukawa charge cancellation 위해).
    """
    return Tensor(
        "chi",
        [s["dirac"].upper(α_name)],
        reps={"SU(2)": "singlet", "U(1)_Y": "-1", "Lorentz": "spinor"},
        statistics="fermionic",
    )


def make_chibar(s, α_name="α"):
    return Tensor(
        "chibar",
        [s["dirac"].lower(α_name)],
        reps={"SU(2)": "singlet", "U(1)_Y": "+1", "Lorentz": "conj_spinor"},
        statistics="fermionic",
    )


def make_W(s, A_name="A", μ_name="μ", ν_name="ν"):
    """$W^A_{\\mu\\nu}$ — SU(2) adj field strength (antisym in μν).

    v1 단순화: Lorentz rep을 ``singlet``으로 두어 ``lorentz_vector_action``의
    single-frame-index 가정을 회피. 수치적으로 $W \\cdot W$가 Lorentz invariant이므로
    $\\delta_{Lorentz}(W \\cdot W) = 0$이라는 결과는 동일. rank-2 vector rep 처리는
    M5+ 작업.
    """
    return Tensor(
        "W",
        [
            s["su2_adj"].upper(A_name),
            s["st"].lower(μ_name),
            s["st"].lower(ν_name),
        ],
        antisymmetric_pairs=[(1, 2)],
        reps={"SU(2)": "adj", "U(1)_Y": "0", "Lorentz": "singlet"},
    )


def make_W_dual(s, A_name="A", μ_name="μ", ν_name="ν"):
    return Tensor(
        "W",
        [
            s["su2_adj"].lower(A_name),
            s["st"].upper(μ_name),
            s["st"].upper(ν_name),
        ],
        antisymmetric_pairs=[(1, 2)],
        reps={"SU(2)": "adj", "U(1)_Y": "0", "Lorentz": "singlet"},
    )


# ─── 1. 멀티-rep Dirac mass term ─────────────────────────────


def test_lepton_mass_invariance(setup_sm):
    """$\\bar\\psi^\\alpha_i \\psi^i_\\alpha$ — 모든 gauge generator에서 invariant.

    SU(2): fund × antifund 직접 contraction.
    U(1)_Y: $-1/2 + 1/2 = 0$.
    Lorentz: spinor × conj_spinor 직접 contraction.
    """
    s = setup_sm
    psi = make_psi(s)
    psibar = make_psibar(s)
    L = TensorProduct(psibar, psi)

    for gen_key in ["su2_gen", "u1y_gen", "lorentz_gen"]:
        delta = apply_generator(L, s[gen_key])
        final = simplify(delta)
        assert isinstance(final, ZeroTensor), f"failed for {gen_key}: {final!r}"


# ─── 2. Higgs potential (mass + quartic) ────────────────────


def test_higgs_quartic_invariance(setup_sm):
    """$(\\bar H H)^2 = \\bar H_i H^i \\bar H_j H^j$ SU(2) × U(1)_Y invariant."""
    s = setup_sm
    Hi = make_H(s, "i"); Hbar_i = make_Hbar(s, "i")
    Hj = make_H(s, "j"); Hbar_j = make_Hbar(s, "j")
    L = TensorProduct(
        TensorProduct(Hbar_i, Hi),
        TensorProduct(Hbar_j, Hj),
    )

    for gen_key in ["su2_gen", "u1y_gen"]:
        delta = apply_generator(L, s[gen_key])
        final = simplify(delta)
        assert isinstance(final, ZeroTensor), f"failed for {gen_key}: {final!r}"


# ─── 3. Yukawa-like 3-field bilinear ─────────────────────────


def test_yukawa_like_invariance(setup_sm):
    """$\\bar\\psi^\\alpha_i H^i \\chi_\\alpha$ — 3-field bilinear의 gauge invariance.

    설정: $\\bar\\psi$ = (antifund SU(2), Y=+1/2, conj_spinor),
          $H$ = (fund SU(2), Y=+1/2, singlet),
          $\\chi_\\alpha$ = (singlet SU(2), Y=$-1$, conj_spinor).

    체크:
      - SU(2): antifund · fund · singlet — i 인덱스 직접 contract, χ는 invariant.
      - U(1)_Y: $+1/2 + 1/2 + (-1) = 0$.
      - Lorentz: conj_spinor · singlet · conj_spinor — α는 \\bar\\psi의 lower와
        χ_lower 사이에 contract... 잠깐. 두 conj_spinor가 같은 spinor 인덱스로
        직접 contract되긴 어렵다 (둘 다 lower라서).

        v1 단순화로 χ를 상위 spinor (rep "spinor", α 위)로 두고 \\bar\\psi의
        lower α와 contract.
    """
    s = setup_sm
    psibar = make_psibar(s, "i", "α")  # antifund i lower, conj_spinor α lower
    H = make_H(s, "i")  # fund i upper
    χ = Tensor(
        "chi",
        [s["dirac"].upper("α")],  # spinor α upper
        reps={"SU(2)": "singlet", "U(1)_Y": "-1", "Lorentz": "spinor"},
        statistics="fermionic",
    )
    L = TensorProduct(psibar, TensorProduct(H, χ))
    # contraction: psibar_i (lo) ↔ H^i (up); psibar_α (lo) ↔ χ^α (up).

    for gen_key in ["su2_gen", "u1y_gen", "lorentz_gen"]:
        delta = apply_generator(L, s[gen_key])
        final = simplify(delta)
        assert isinstance(final, ZeroTensor), f"failed for {gen_key}: {final!r}"


# ─── 4. SU(2) Yang-Mills (M2.5 재현) ────────────────────────


def test_su2_yang_mills_invariance(setup_sm):
    """$-\\tfrac14 W^A_{\\mu\\nu} W_A^{\\mu\\nu}$ SU(2) gauge invariance (M2.5 패턴)."""
    s = setup_sm
    W1 = make_W(s, "A", "μ", "ν")
    W2 = make_W_dual(s, "A", "μ", "ν")
    L = ScalarMul(-0.25, TensorProduct(W1, W2))

    delta = apply_generator(L, s["su2_gen"])
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


# ─── 5. 통합 SM-lite 라그랑지안 ──────────────────────────────


def test_sm_lite_full_lagrangian(setup_sm):
    """모든 항을 합쳐 한 라그랑지안으로: 각 generator마다 합쳐서 0.

    $\\mathcal L = \\bar\\psi\\psi + \\bar H H + (\\bar H H)^2 + \\bar\\psi^i H_i \\chi
                  -\\tfrac14 W^A W_A$
    """
    s = setup_sm

    # Dirac mass-like
    psi = make_psi(s, "i", "α")
    psibar = make_psibar(s, "i", "α")
    L_lepton = TensorProduct(psibar, psi)

    # Higgs mass + quartic — 별도 dummy 인덱스 (k, l)로 충돌 회피
    Hk = make_H(s, "k"); Hbar_k = make_Hbar(s, "k")
    Hl = make_H(s, "l"); Hbar_l = make_Hbar(s, "l")
    L_higgs_mass = TensorProduct(Hbar_k, Hk)
    L_higgs_quartic = TensorProduct(
        TensorProduct(Hbar_k, Hk),
        TensorProduct(Hbar_l, Hl),
    )

    # Yukawa-like (별도 인덱스 사용 권장 — 충돌 방지)
    psibar_yuk = make_psibar(s, "m", "β")
    H_yuk = make_H(s, "m")
    χ = Tensor(
        "chi",
        [s["dirac"].upper("β")],
        reps={"SU(2)": "singlet", "U(1)_Y": "-1", "Lorentz": "spinor"},
        statistics="fermionic",
    )
    L_yukawa = TensorProduct(psibar_yuk, TensorProduct(H_yuk, χ))

    # Yang-Mills (별도 인덱스 — adj A, spacetime ν λ)
    W1 = make_W(s, "A", "ν", "λ")
    W2 = make_W_dual(s, "A", "ν", "λ")
    L_ym = ScalarMul(-0.25, TensorProduct(W1, W2))

    # 통합
    L = TensorSum(
        L_lepton,
        TensorSum(
            L_higgs_mass,
            TensorSum(
                L_higgs_quartic,
                TensorSum(L_yukawa, L_ym),
            ),
        ),
    )

    for gen_key in ["su2_gen", "u1y_gen", "lorentz_gen"]:
        delta = apply_generator(L, s[gen_key])
        final = simplify(delta)
        assert isinstance(final, ZeroTensor), (
            f"FAILED for {gen_key}: got {type(final).__name__}: {final!r}"
        )
