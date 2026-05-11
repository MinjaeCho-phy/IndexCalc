"""Tensor 속성 슬롯 (symmetric_pairs / traceless / transverse) acceptance.

2026-05-11 정책 전환에 따라 Tensor.__init__에 추가된 슬롯들이
- 옳게 저장되고
- validation이 올바르게 동작함
을 검증한다. Simplification rule(γ^ij·h^TT_ij → 0 등)은 별도 단계에서 처리.
"""

import pytest

from indexcalc import IndexSpace, Tensor


@pytest.fixture
def sp_space():
    return IndexSpace("sp", dim=3, indices="ijklmn", metric="γ")


@pytest.fixture
def st_space():
    return IndexSpace("st", dim=4, indices="μνρσλ", metric="g")


# ─── 기본 저장 ───────────────────────────────────────────────────

def test_symmetric_pairs_stored(sp_space):
    i, j = sp_space.lower("i"), sp_space.lower("j")
    g = Tensor("g", [i, j], symmetric_pairs=[(0, 1)])
    assert g.symmetric_pairs == ((0, 1),)
    assert g.antisymmetric_pairs == ()


def test_traceless_and_transverse_stored(sp_space):
    i, j = sp_space.lower("i"), sp_space.lower("j")
    h = Tensor(
        "h^TT", [i, j],
        symmetric_pairs=[(0, 1)],
        traceless=[(0, 1)],
        transverse=[0, 1],
    )
    assert h.symmetric_pairs == ((0, 1),)
    assert h.traceless == ((0, 1),)
    assert h.transverse == (0, 1)


def test_vector_transverse_only(sp_space):
    i = sp_space.lower("i")
    B = Tensor("B^V", [i], transverse=[0])
    assert B.transverse == (0,)
    assert B.symmetric_pairs == ()
    assert B.traceless == ()


def test_default_slots_empty(sp_space):
    i, j = sp_space.lower("i"), sp_space.lower("j")
    T = Tensor("T", [i, j])
    assert T.symmetric_pairs == ()
    assert T.traceless == ()
    assert T.transverse == ()


# ─── validation ──────────────────────────────────────────────────

def test_reject_sym_antisym_overlap(sp_space):
    i, j = sp_space.lower("i"), sp_space.lower("j")
    with pytest.raises(ValueError, match="both symmetric and antisymmetric"):
        Tensor(
            "X", [i, j],
            antisymmetric_pairs=[(0, 1)],
            symmetric_pairs=[(0, 1)],
        )


def test_reject_traceless_crossing_spaces(sp_space, st_space):
    """traceless slot 쌍은 같은 IndexSpace여야 한다 (γ^ij 같은 metric 필요)."""
    i = sp_space.lower("i")
    μ = st_space.lower("μ")
    with pytest.raises(ValueError, match="crosses different IndexSpaces"):
        Tensor("X", [i, μ], traceless=[(0, 1)])


def test_reject_out_of_range_transverse(sp_space):
    i = sp_space.lower("i")
    with pytest.raises(ValueError, match="invalid transverse slot"):
        Tensor("B", [i], transverse=[5])


def test_reject_self_paired_symmetric(sp_space):
    i, j = sp_space.lower("i"), sp_space.lower("j")
    with pytest.raises(ValueError, match="invalid symmetric_pair"):
        Tensor("X", [i, j], symmetric_pairs=[(0, 0)])


# ─── pair normalization ─────────────────────────────────────────

def test_pair_order_normalized(sp_space):
    i, j = sp_space.lower("i"), sp_space.lower("j")
    h = Tensor("h", [i, j], symmetric_pairs=[(1, 0)])
    assert h.symmetric_pairs == ((0, 1),)
