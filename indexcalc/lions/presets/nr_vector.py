"""NR vector preset — Φ^i SO(N)/O(N) vector field + canonical Lagrangians.

Smallest non-trivial NR mechanics target for v2 ML training: scalar
vector fields under spatial rotation group. Term catalog includes:

    δ_ij \\dot Φ^i \\dot Φ^j   (kinetic)
    δ_ij Φ^i Φ^j              (mass / harmonic)
    (Φ^2)^2                    (quartic)
    ε_{ijk} Φ^i Ψ^j Ξ^k        (SO(N) ✓, O(N) ✗)
    f(Φ^2)                     (non-polynomial via ScalarFunction)

This preset returns the index space, groups (O(N)/SO(N)), generators,
and a LIONS ``FieldRegistry`` so the v2 enumerator and labeler can wire
up without rebuilding fixtures across tests/scripts.
"""

from __future__ import annotations
from dataclasses import dataclass

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum, ScalarMul
from indexcalc.core.group import Group
from indexcalc.core.generator import Generator, make_o_n_generator
from indexcalc.core.scalar_function import ScalarFunction
from indexcalc.adm import TimeDeriv
from indexcalc.lions.fields import FieldSpec, FieldRegistry, SlotSpec


@dataclass
class NRVectorSetup:
    """NR vector preset의 모든 fixture를 묶어 caller에게 노출."""
    vec: IndexSpace
    o_group: Group
    so_group: Group
    o_gen: Generator
    so_gen: Generator
    fields: FieldRegistry


def build_nr_vector(N: int = 3) -> NRVectorSetup:
    """N-dim NR vector preset. 기본 N=3 (Kepler-스코프).

    Returns
    -------
    NRVectorSetup
        - ``vec`` IndexSpace (이름 ``"soN_vec"``, dim=N, metric ``"delta"``)
        - O(N), SO(N) Group + generator
        - FieldRegistry: Φ, Ψ, A, B, C (모두 vector) + Σ (singlet, 외부 source)
    """
    vec = IndexSpace(
        f"so{N}_vec", dim=N,
        indices="ijklmnpqrstuv", metric="delta",
    )

    o_n = Group(f"O({N})", dim=N * (N - 1) // 2, abelian=False)
    o_n.add_rep("vector", dim=N)
    o_n.add_rep("singlet", dim=1)

    so_n = Group(f"SO({N})", dim=N * (N - 1) // 2, abelian=False)
    so_n.add_rep("vector", dim=N)
    so_n.add_rep("singlet", dim=1)

    o_gen = make_o_n_generator(o_n, vec)
    so_gen = make_o_n_generator(so_n, vec)

    reg = FieldRegistry()
    common_reps = {f"O({N})": "vector", f"SO({N})": "vector"}
    for name in ("Phi", "Psi", "A", "B", "C"):
        reg.add(FieldSpec(
            name=name,
            slots=(SlotSpec(vec, "upper"),),
            reps=dict(common_reps),
            mass_dim=1.0,
            statistics="bosonic",
        ))
    # 외부 singlet source (force term 등)
    reg.add(FieldSpec(
        name="Sigma",
        slots=(),
        reps={f"O({N})": "singlet", f"SO({N})": "singlet"},
        mass_dim=2.0,
        statistics="bosonic",
    ))

    return NRVectorSetup(
        vec=vec, o_group=o_n, so_group=so_n,
        o_gen=o_gen, so_gen=so_gen, fields=reg,
    )


# ─── Canonical Lagrangian builders ──────────────────────


def _delta_lower(vec: IndexSpace, i: str, j: str, reps: dict) -> Tensor:
    return Tensor(
        "delta", [vec.lower(i), vec.lower(j)],
        symmetric_pairs=[(0, 1)],
        reps=reps,
    )


def kinetic_term(setup: NRVectorSetup, field_name: str = "Phi") -> TensorExpr:
    """(1/2) δ_ij \\dot Φ^i \\dot Φ^j."""
    reps = setup.fields.get(field_name).reps
    Phi_i = Tensor(field_name, [setup.vec.upper("i")], reps=reps)
    Phi_j = Tensor(field_name, [setup.vec.upper("j")], reps=reps)
    delta = _delta_lower(
        setup.vec, "i", "j",
        {k: "singlet" for k in reps},
    )
    return ScalarMul(
        0.5,
        delta * TimeDeriv(Phi_i) * TimeDeriv(Phi_j),
    )


def mass_term(setup: NRVectorSetup, field_name: str = "Phi") -> TensorExpr:
    """(1/2) δ_ij Φ^i Φ^j."""
    reps = setup.fields.get(field_name).reps
    Phi_i = Tensor(field_name, [setup.vec.upper("i")], reps=reps)
    Phi_j = Tensor(field_name, [setup.vec.upper("j")], reps=reps)
    delta = _delta_lower(
        setup.vec, "i", "j",
        {k: "singlet" for k in reps},
    )
    return ScalarMul(0.5, delta * Phi_i * Phi_j)


def quartic_term(setup: NRVectorSetup, field_name: str = "Phi") -> TensorExpr:
    """(Φ^2)^2 = (δ_ij Φ^i Φ^j) · (δ_kl Φ^k Φ^l)."""
    reps = setup.fields.get(field_name).reps
    Phi_i = Tensor(field_name, [setup.vec.upper("i")], reps=reps)
    Phi_j = Tensor(field_name, [setup.vec.upper("j")], reps=reps)
    Phi_k = Tensor(field_name, [setup.vec.upper("k")], reps=reps)
    Phi_l = Tensor(field_name, [setup.vec.upper("l")], reps=reps)
    singlet_reps = {k: "singlet" for k in reps}
    d_ij = _delta_lower(setup.vec, "i", "j", singlet_reps)
    d_kl = _delta_lower(setup.vec, "k", "l", singlet_reps)
    return d_ij * Phi_i * Phi_j * d_kl * Phi_k * Phi_l


def inverse_sqrt_potential(setup: NRVectorSetup,
                           field_name: str = "Phi") -> TensorExpr:
    """κ / sqrt(δ_kl Φ^k Φ^l) — Kepler / Coulomb 형식."""
    reps = setup.fields.get(field_name).reps
    Phi_k = Tensor(field_name, [setup.vec.upper("k")], reps=reps)
    Phi_l = Tensor(field_name, [setup.vec.upper("l")], reps=reps)
    singlet_reps = {k: "singlet" for k in reps}
    d_kl = _delta_lower(setup.vec, "k", "l", singlet_reps)
    r_sq = d_kl * Phi_k * Phi_l
    return ScalarFunction("inv_sqrt", r_sq)


def epsilon_trilinear(setup: NRVectorSetup,
                      f1: str = "A", f2: str = "B", f3: str = "C") -> TensorExpr:
    """ε_{ijk} A^i B^j C^k — SO(N) ✓ O(N) ✗ (N=3 한정)."""
    N = setup.vec.dim
    if N != 3:
        raise ValueError(f"epsilon_trilinear assumes N=3, got N={N}")
    reps1 = setup.fields.get(f1).reps
    reps2 = setup.fields.get(f2).reps
    reps3 = setup.fields.get(f3).reps
    A = Tensor(f1, [setup.vec.upper("i")], reps=reps1)
    B = Tensor(f2, [setup.vec.upper("j")], reps=reps2)
    C = Tensor(f3, [setup.vec.upper("k")], reps=reps3)
    eps = Tensor(
        "epsilon",
        [setup.vec.lower("i"), setup.vec.lower("j"), setup.vec.lower("k")],
        antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
    )
    return eps * A * B * C


def broken_rotation_term(setup: NRVectorSetup) -> TensorExpr:
    """Hard-negative: vector × wrong-rep singlet (rotation 깨짐)."""
    Phi_reps = setup.fields.get("Phi").reps
    # Sigma는 이미 singlet — 직접 vector index contract 불가, 다른 부정확
    # 패턴으로 break term 만들기: 자체 fake singlet에 vector index 강제.
    Phi = Tensor("Phi", [setup.vec.upper("i")], reps=Phi_reps)
    fake = Tensor(
        "FakeSinglet", [setup.vec.lower("i")],
        reps={k: "singlet" for k in Phi_reps},
    )
    return Phi * fake


# 타입 힌트
from indexcalc.core.tensor import TensorExpr  # noqa: E402 (placed at bottom
# to avoid circular ref in build_nr_vector signature)
