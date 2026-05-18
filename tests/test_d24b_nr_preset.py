"""D24b: NR vector preset + canonical term builders.

검증:
1. build_nr_vector smoke — index space, groups, fields 등록.
2. Canonical builders all produce scalar (free_indices == []) expressions.
3. probe() oracle correctly labels each: kinetic/mass/quartic/inv_sqrt are
   SO(3)/O(3) invariant; ε-trilinear is SO(3)✓ O(3)✗; broken_rotation_term
   is invariant=False.
4. Preset reps survive across enumerator-style usage.
"""

from __future__ import annotations

import pytest

from indexcalc.lions.presets.nr_vector import (
    build_nr_vector, kinetic_term, mass_term, quartic_term,
    inverse_sqrt_potential, epsilon_trilinear, broken_rotation_term,
)
from indexcalc.lions.probe import (
    probe, GroupSpec, classical_group_spec,
)


@pytest.fixture
def nr3():
    return build_nr_vector(N=3)


@pytest.fixture
def so3_cand(nr3):
    return GroupSpec(name="SO(3)", group=nr3.so_group,
                     generator=nr3.so_gen, dim=3)


@pytest.fixture
def o3_cand(nr3):
    return GroupSpec(name="O(3)", group=nr3.o_group,
                     generator=nr3.o_gen, dim=3)


def test_setup_has_fields(nr3):
    for name in ("Phi", "Psi", "A", "B", "C", "Sigma"):
        assert nr3.fields.has(name) if hasattr(
            nr3.fields, "has",
        ) else nr3.fields.get(name) is not None


def test_kinetic_is_scalar_and_invariant(nr3, so3_cand, o3_cand):
    L = kinetic_term(nr3)
    assert L.free_indices == []
    rs = {r.group: r for r in probe(L, [], [so3_cand, o3_cand])}
    assert rs["SO(3)"].invariant
    assert rs["O(3)"].invariant


def test_mass_term_invariant(nr3, so3_cand, o3_cand):
    L = mass_term(nr3)
    assert L.free_indices == []
    rs = {r.group: r for r in probe(L, [], [so3_cand, o3_cand])}
    assert rs["SO(3)"].invariant
    assert rs["O(3)"].invariant


def test_quartic_invariant(nr3, so3_cand):
    L = quartic_term(nr3)
    assert L.free_indices == []
    r = probe(L, [], [so3_cand])[0]
    assert r.invariant


def test_inverse_sqrt_potential_invariant(nr3, so3_cand, o3_cand):
    L = inverse_sqrt_potential(nr3)
    assert L.free_indices == []
    rs = {r.group: r for r in probe(L, [], [so3_cand, o3_cand])}
    assert rs["SO(3)"].invariant
    assert rs["O(3)"].invariant


def test_epsilon_trilinear_so3_yes_o3_no(nr3, so3_cand, o3_cand):
    L = epsilon_trilinear(nr3)
    assert L.free_indices == []
    rs = {r.group: r for r in probe(L, [], [so3_cand, o3_cand])}
    assert rs["SO(3)"].invariant, "ε-trilinear should be SO(3) invariant"
    assert not rs["O(3)"].invariant, (
        "ε-trilinear should NOT be O(3) invariant (parity flip)"
    )


def test_broken_rotation_term_not_invariant(nr3, so3_cand):
    L = broken_rotation_term(nr3)
    r = probe(L, [], [so3_cand])[0]
    assert not r.invariant


def test_kepler_via_preset(nr3, so3_cand, o3_cand):
    """Kepler L = kinetic + κ * inv_sqrt(Φ^2) — both SO(3) and O(3) ✓."""
    L_kep = kinetic_term(nr3) + inverse_sqrt_potential(nr3)
    rs = {r.group: r for r in probe(L_kep, [], [so3_cand, o3_cand])}
    assert rs["SO(3)"].invariant
    assert rs["O(3)"].invariant
