"""G4: Sym / Antisym / TraceFreeSym 노드 acceptance.

2026-05-11 정책 전환에 따라 표현식 트리에 명시적 대칭화 연산자를 추가.
첫 패스: n=2 swap, prefix LaTeX, expand 함수.
"""

import pytest

from indexcalc import (
    IndexSpace, Tensor, MetricRegistry, PartialDeriv,
    Sym, Antisym, TraceFreeSym, expand_symmetrization,
)
from indexcalc.core.index import Index


@pytest.fixture
def sp_space():
    return IndexSpace("sp", dim=3, indices="ijklmn", metric="γ")


@pytest.fixture
def metrics(sp_space):
    reg = MetricRegistry()
    i_lo, j_lo = sp_space.lower("i"), sp_space.lower("j")
    i_up, j_up = sp_space.upper("i"), sp_space.upper("j")
    reg.register(
        Tensor("γ", [i_lo, j_lo], symmetric_pairs=[(0, 1)]),
        Tensor("γ", [i_up, j_up], symmetric_pairs=[(0, 1)]),
        sp_space,
    )
    return reg


# ─── 노드 생성 ──────────────────────────────────────────────────

def test_sym_node_preserves_free_indices(sp_space):
    i, j = sp_space.lower("i"), sp_space.lower("j")
    T = Tensor("T", [i, j])
    s = Sym(T, [i, j])
    assert {idx.name for idx in s.free_indices} == {"i", "j"}


def test_antisym_node_repr(sp_space):
    i, j = sp_space.lower("i"), sp_space.lower("j")
    T = Tensor("T", [i, j])
    a = Antisym(T, [i, j])
    assert "Antisym" in repr(a) and "i" in repr(a) and "j" in repr(a)


def test_reject_cross_space(sp_space):
    """sym_indices 두 개는 같은 IndexSpace여야."""
    st = IndexSpace("st", dim=4, indices="μν", metric="g")
    i = sp_space.lower("i")
    μ = st.lower("μ")
    T = Tensor("T", [i, μ])
    with pytest.raises(ValueError, match="share IndexSpace"):
        Sym(T, [i, μ])


def test_reject_n_not_2(sp_space):
    i, j, k = sp_space.lower("i"), sp_space.lower("j"), sp_space.lower("k")
    T = Tensor("T", [i, j, k])
    with pytest.raises(NotImplementedError, match="n=2 indices only"):
        Sym(T, [i, j, k])


def test_reject_index_not_in_expr(sp_space):
    """sym_indices가 expr의 free 안에 있어야."""
    i, j, k = sp_space.lower("i"), sp_space.lower("j"), sp_space.lower("k")
    T = Tensor("T", [i, j])
    with pytest.raises(ValueError, match="not free"):
        Sym(T, [i, k])


# ─── expand: n=2 case ────────────────────────────────────────────

def test_expand_sym_pair_tensor(sp_space):
    """Sym(T_{ij}, [i,j]) → ½(T_{ij} + T_{ji})"""
    i, j = sp_space.lower("i"), sp_space.lower("j")
    T = Tensor("T", [i, j])
    expanded = expand_symmetrization(Sym(T, [i, j]))
    latex = expanded.to_latex()
    assert "T_{i j}" in latex and "T_{j i}" in latex
    assert "\\frac{1}{2}" in latex
    assert " + " in latex


def test_expand_antisym_pair_tensor(sp_space):
    """Antisym(T_{ij}, [i,j]) → ½(T_{ij} − T_{ji})"""
    i, j = sp_space.lower("i"), sp_space.lower("j")
    T = Tensor("T", [i, j])
    expanded = expand_symmetrization(Antisym(T, [i, j]))
    latex = expanded.to_latex()
    assert "T_{i j}" in latex and "T_{j i}" in latex
    assert " - " in latex


def test_expand_sym_partial_vector(sp_space):
    """Sym(∂_i V_j, [i,j]) → ½(∂_i V_j + ∂_j V_i) — D_(i E_j) 패턴"""
    i, j = sp_space.lower("i"), sp_space.lower("j")
    V = Tensor("V", [j])
    expr = PartialDeriv(V, i)
    expanded = expand_symmetrization(Sym(expr, [i, j]))
    latex = expanded.to_latex()
    # 두 항: ∂_i V_j 와 ∂_j V_i 모두 등장
    assert "\\partial_{i} V_{j}" in latex
    assert "\\partial_{j} V_{i}" in latex


def test_expand_tracefreesym_needs_metric(sp_space):
    i, j = sp_space.lower("i"), sp_space.lower("j")
    T = Tensor("T", [i, j])
    with pytest.raises(ValueError, match="MetricRegistry"):
        expand_symmetrization(TraceFreeSym(T, [i, j]))


def test_expand_tracefreesym_basic(sp_space, metrics):
    """TraceFreeSym(T_{ij}, [i,j]) — 결과에 γ_{ij}, γ^{kl}, T 들이 모두 등장."""
    i, j = sp_space.lower("i"), sp_space.lower("j")
    T = Tensor("T", [i, j])
    expanded = expand_symmetrization(TraceFreeSym(T, [i, j]), metrics)
    latex = expanded.to_latex()
    # symmetric part
    assert "T_{i j}" in latex
    assert "T_{j i}" in latex
    # trace subtraction
    assert "\\gamma_{i j}" in latex
    assert "\\gamma^{" in latex


# ─── LaTeX display ───────────────────────────────────────────────

def test_display_sym_node(sp_space):
    i, j = sp_space.lower("i"), sp_space.lower("j")
    T = Tensor("T", [i, j])
    out = Sym(T, [i, j]).to_latex()
    assert "\\mathrm{Sym}" in out
    assert "T_{i j}" in out


def test_display_antisym_node(sp_space):
    i, j = sp_space.lower("i"), sp_space.lower("j")
    T = Tensor("T", [i, j])
    out = Antisym(T, [i, j]).to_latex()
    assert "\\mathrm{Antisym}" in out


def test_display_tracefreesym_node(sp_space):
    i, j = sp_space.lower("i"), sp_space.lower("j")
    T = Tensor("T", [i, j])
    out = TraceFreeSym(T, [i, j]).to_latex()
    assert "\\mathrm{TFS}" in out
