"""LIONS M9 acceptance — chiral projectors $P_L, P_R, \\gamma_5$ IR primitives
+ Clifford anticommute $\\{\\gamma_5, \\gamma^\\mu\\} = 0$.

신규 도구:

- ``apply_chiral_projector_identities``: 인접 (chain-wise) projector/γ_5
  contraction 을 identity table 에 따라 rewrite.

  | left.col → right.row | result |
  |---|---|
  | P_L · P_L | P_L |
  | P_R · P_R | P_R |
  | P_L · P_R | 0 |
  | P_R · P_L | 0 |
  | γ_5 · P_L, P_L · γ_5 | P_L |
  | γ_5 · P_R, P_R · γ_5 | -P_R |
  | γ_5 · γ_5 | δ_spinor (invariant tying outer slots) |

- ``apply_gamma5_gamma_anticommute``: ``γ_5 γ^μ → -γ^μ γ_5`` — γ_5 를
  γ chain 의 오른쪽 끝으로 정규화.

설계 노트: `notes/m9_chiral_projectors.md` (LIONS repo).
"""

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.simplify import simplify


# ─── Helpers ────────────────────────────────────────────────


@pytest.fixture
def spinor_space():
    return IndexSpace("dirac", dim=4, indices="αβγδερστυφ")


@pytest.fixture
def frame_space():
    return IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")


def make_psi_bar(sp, alpha="α"):
    """ψ̄ — Dirac conj_spinor, lower spinor index."""
    return Tensor(
        "psibar", [sp.lower(alpha)],
        reps={"Lorentz": "conj_spinor"},
        statistics="fermionic",
    )


def make_psi(sp, alpha="α"):
    """ψ — Dirac spinor, upper spinor index."""
    return Tensor(
        "psi", [sp.upper(alpha)],
        reps={"Lorentz": "spinor"},
        statistics="fermionic",
    )


def make_P_L(sp, up="α", down="β"):
    return Tensor("P_L", [sp.upper(up), sp.lower(down)], reps={})


def make_P_R(sp, up="α", down="β"):
    return Tensor("P_R", [sp.upper(up), sp.lower(down)], reps={})


def make_gamma_5(sp, up="α", down="β"):
    return Tensor("gamma_5", [sp.upper(up), sp.lower(down)], reps={})


def make_gamma(frame, sp, vec="μ", up="α", down="β"):
    """γ^μ{}^α{}_β."""
    return Tensor(
        "gamma",
        [frame.upper(vec), sp.upper(up), sp.lower(down)],
        reps={},
    )


def _strip_scalar(expr):
    """ScalarMul wrapper 벗기고 (scalar, body) 반환. ScalarMul 없으면 (1, expr)."""
    if isinstance(expr, ScalarMul):
        return expr.scalar, expr.expr
    return 1, expr


# ─── 1. P_L^2 = P_L ─────────────────────────────────────────


def test_PL_squared_collapses_to_PL(spinor_space):
    """P_L^α{}_β P_L^β{}_γ → P_L^α{}_γ."""
    sp = spinor_space
    PL_1 = make_P_L(sp, "α", "β")
    PL_2 = make_P_L(sp, "β", "γ")
    expr = TensorProduct(PL_1, PL_2)

    out = simplify(expr)
    assert isinstance(out, Tensor)
    assert out.name == "P_L"
    free = out.free_indices
    assert len(free) == 2
    names = {idx.name for idx in free}
    assert names == {"α", "γ"}


# ─── 2. P_R^2 = P_R ─────────────────────────────────────────


def test_PR_squared_collapses_to_PR(spinor_space):
    sp = spinor_space
    PR_1 = make_P_R(sp, "α", "β")
    PR_2 = make_P_R(sp, "β", "γ")
    expr = TensorProduct(PR_1, PR_2)

    out = simplify(expr)
    assert isinstance(out, Tensor)
    assert out.name == "P_R"


# ─── 3. P_L P_R = 0, P_R P_L = 0 ───────────────────────────


def test_PL_PR_vanishes(spinor_space):
    sp = spinor_space
    expr = TensorProduct(make_P_L(sp, "α", "β"), make_P_R(sp, "β", "γ"))

    out = simplify(expr)
    assert isinstance(out, ZeroTensor)


def test_PR_PL_vanishes(spinor_space):
    sp = spinor_space
    expr = TensorProduct(make_P_R(sp, "α", "β"), make_P_L(sp, "β", "γ"))

    out = simplify(expr)
    assert isinstance(out, ZeroTensor)


# ─── 4. γ_5 P_L = P_L, γ_5 P_R = -P_R ──────────────────────


def test_gamma5_PL_collapses_to_PL(spinor_space):
    sp = spinor_space
    expr = TensorProduct(make_gamma_5(sp, "α", "β"), make_P_L(sp, "β", "γ"))

    out = simplify(expr)
    scalar, body = _strip_scalar(out)
    assert scalar == 1
    assert isinstance(body, Tensor) and body.name == "P_L"


def test_gamma5_PR_collapses_to_minus_PR(spinor_space):
    sp = spinor_space
    expr = TensorProduct(make_gamma_5(sp, "α", "β"), make_P_R(sp, "β", "γ"))

    out = simplify(expr)
    scalar, body = _strip_scalar(out)
    assert scalar == -1
    assert isinstance(body, Tensor) and body.name == "P_R"


# ─── 5. {γ_5, γ^μ} = 0  (anticommute)  ──────────────────────


def test_gamma5_gamma_anticommute_pushes_right(spinor_space, frame_space):
    """γ_5^α{}_β γ^{μ,β}{}_γ → -γ^{μ,α}{}_ρ γ_5^ρ{}_γ.

    Result: ScalarMul(-1, TensorProduct(γ, γ_5)) with γ_5 on the right.
    """
    sp, ft = spinor_space, frame_space
    g5 = make_gamma_5(sp, "α", "β")
    g = make_gamma(ft, sp, vec="μ", up="β", down="γ")
    expr = TensorProduct(g5, g)

    out = simplify(expr)
    scalar, body = _strip_scalar(out)
    assert scalar == -1, f"expected -1, got {scalar}"

    # body should be γ · γ_5 (γ_5 pushed to the right)
    assert isinstance(body, TensorProduct)
    # γ comes first now
    left, right = body.left, body.right
    assert isinstance(left, Tensor) and left.name == "gamma"
    assert isinstance(right, Tensor) and right.name == "gamma_5"

    # Outer indices preserved: γ.row = α (was γ_5.row), γ_5.col = γ (was γ.col)
    assert left.indices[1].name == "α"
    assert right.indices[1].name == "γ"


# ─── 6. γ_5 γ^μ γ_5 = -γ^μ  (end-to-end: anticommute + γ_5^2=1) ─


def test_gamma5_gamma_gamma5_equals_minus_gamma(spinor_space, frame_space):
    """γ_5^α{}_β γ^{μ,β}{}_γ γ_5^γ{}_δ → -γ^{μ,α}{}_δ.

    Step 1 (anticommute): push left γ_5 past γ → -γ γ_5 γ_5.
    Step 2 (γ_5^2 = δ): γ_5 γ_5 → δ_spinor, then δ collapse with γ via dummy.

    Net: ScalarMul(-1, gamma) with outer indices (μ↑, α↑, δ↓).
    """
    sp, ft = spinor_space, frame_space
    g5_a = make_gamma_5(sp, "α", "β")
    g = make_gamma(ft, sp, vec="μ", up="β", down="γ")
    g5_b = make_gamma_5(sp, "γ", "δ")
    expr = TensorProduct(g5_a, TensorProduct(g, g5_b))

    out = simplify(expr)
    scalar, body = _strip_scalar(out)
    assert scalar == -1, f"expected -1, got {scalar}: {out!r}"

    # body should contain γ tensor (possibly contracted with δ_spinor invariant,
    # but the free indices must reduce to (μ↑, α↑, δ↓)).
    free = body.free_indices
    free_names = {(idx.name, idx.position) for idx in free}
    assert ("μ", "upper") in free_names
    assert ("α", "upper") in free_names
    assert ("δ", "lower") in free_names
