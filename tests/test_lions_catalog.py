"""LIONS v2.5 M1 — catalog smoke test.

Coverage:
- Catalog has exactly 19 entries (per v2_5_redirect decisions).
- Every label is unique.
- Each family bucket has the right shape (classical 4 entries × 4 N, Lorentz/Poincaré singletons, U(1) singleton).
- ``build_groupspec`` materialises every entry without raising.
- Built ``GroupSpec`` carries the expected rep set for each family.
"""

from __future__ import annotations
import pytest

from indexcalc.lions.catalog import (
    CATALOG, CatalogEntry, all_labels, get, build_groupspec,
)


# ─── Shape ──────────────────────────────────────────────


def test_catalog_size_is_26():
    assert len(CATALOG) == 26


def test_labels_are_unique():
    labels = all_labels()
    assert len(labels) == len(set(labels))


@pytest.mark.parametrize("group_name,expected_Ns", [
    ("U(N)",  (2, 3, 4, 5)),
    ("SU(N)", (2, 3, 4, 5)),
    ("O(N)",  (2, 3, 4, 5)),
    ("SO(N)", (2, 3, 4, 5)),
    ("Sp(2N)", (2, 3, 4, 5)),
])
def test_each_classical_family_has_4_entries(group_name, expected_Ns):
    matching = [e for e in CATALOG if e.group_name == group_name]
    assert tuple(e.N for e in matching) == expected_Ns


def test_sp_labels_use_2n_dimension():
    """Sp 엔트리는 rank N을 저장하지만 concrete label은 Sp(2N)."""
    sp = [e for e in CATALOG if e.family == "symplectic"]
    assert [e.label for e in sp] == ["Sp(4)", "Sp(6)", "Sp(8)", "Sp(10)"]
    assert get("Sp(4)").N == 2  # rank 2 → dim 4
    assert "omega" in get("Sp(4)").invariants
    assert "delta" not in get("Sp(4)").invariants


def test_singletons():
    fam = lambda f: [e for e in CATALOG if e.family == f]
    assert len(fam("abelian")) == 1
    assert len(fam("lorentz")) == 1
    assert len(fam("poincare")) == 1
    assert fam("abelian")[0].label == "U(1)"
    assert fam("lorentz")[0].label == "Lorentz"
    assert fam("poincare")[0].label == "Poincare"


def test_get_lookup():
    assert get("U(3)").N == 3
    assert get("SO(4)").family == "orthogonal"
    assert get("Lorentz").family == "lorentz"
    assert get("Sp(6)").family == "symplectic"
    with pytest.raises(KeyError):
        get("Sp(12)")  # rank 6 — out of first round (N=2..5)


# ─── Invariant tensor table ─────────────────────────────


def test_so_n_has_epsilon_o_n_does_not():
    """SO(N)는 ε_{i1..iN} 가짐, O(N)는 reflection으로 부호 flip → 없음."""
    so3 = get("SO(3)")
    o3 = get("O(3)")
    assert "epsilon" in so3.invariants
    assert "epsilon" not in o3.invariants
    assert "delta" in so3.invariants
    assert "delta" in o3.invariants


def test_su_n_has_epsilon_u_n_does_not():
    """SU(N)는 ε_{i1..iN} (N-index totally antisym), U(N)는 없음."""
    assert "epsilon" in get("SU(3)").invariants
    assert "epsilon" not in get("U(3)").invariants


# ─── build_groupspec ────────────────────────────────────


@pytest.mark.parametrize("label", [
    "U(1)",
    "U(2)", "U(3)", "U(4)", "U(5)",
    "SU(2)", "SU(3)", "SU(4)", "SU(5)",
    "O(2)", "O(3)", "O(4)", "O(5)",
    "SO(2)", "SO(3)", "SO(4)", "SO(5)",
    "Sp(4)", "Sp(6)", "Sp(8)", "Sp(10)",
    "SO(2,2)", "SO(3,2)", "SO(4,2)",
    "Lorentz", "Poincare",
])
def test_build_groupspec_works_for_every_entry(label):
    e = get(label)
    spec = build_groupspec(e, prefix=f"t_{label}_")
    assert spec.name in (label, "Lorentz", "Poincare", "U(1)")
    assert spec.generator is not None


def test_sp_groupspec_has_vector_singlet_and_correct_dim():
    spec = build_groupspec(get("Sp(4)"), prefix="t_")
    for r in ("singlet", "vector"):
        assert spec.group.has_rep(r), f"Sp(4) missing rep {r}"
    assert spec.group.get_rep("vector").dim == 4  # 2·rank
    assert spec.dim == 10  # Sp(4) = rank·(2·rank+1) = 2·5


def test_classical_groupspec_carries_supported_reps():
    spec = build_groupspec(get("SU(3)"), prefix="t_")
    for r in ("singlet", "fund", "antifund", "adj"):
        assert spec.group.has_rep(r), f"SU(3) missing rep {r}"


def test_lorentz_groupspec_has_dirac_and_weyl():
    spec = build_groupspec(get("Lorentz"), prefix="t_")
    for r in ("singlet", "vector", "spinor", "L_spinor", "R_spinor"):
        assert spec.group.has_rep(r), f"Lorentz missing rep {r}"


def test_poincare_dim_extends_lorentz_by_4():
    """First round: Poincaré dim = Lorentz dim + 4 translation params."""
    L = build_groupspec(get("Lorentz"), prefix="tL_")
    P = build_groupspec(get("Poincare"), prefix="tP_")
    assert P.dim == L.dim + 4
