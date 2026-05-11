"""scripts/adm_demo.py — Backend 1.b ADM 3+1 분해 demo.

ADM의 lapse N, shift N^i, spatial metric h_{ij}, h^{ij}, extrinsic
curvature K_{ij}를 IndexCalc로 셋업하고:
    - extrinsic_curvature_definition: K_{ij} = (1/(2N))(∂_t h_{ij} - D_i N_j - D_j N_i)
    - K_trace_definition: K = h^{ij} K_{ij}
    - metric_lower/upper_components: g_{μν}, g^{μν} ADM 컴포넌트 dict
출력 형태를 확인한다.
"""

from __future__ import annotations

from indexcalc import (
    IndexSpace, MetricRegistry, LeviCivitaConnection, Tensor,
    ADMSetup, TimeDeriv, LieDeriv, expand_lie_deriv, slice_decompose,
    extrinsic_curvature_definition, K_trace_definition,
    metric_lower_components, metric_upper_components,
    hamiltonian_constraint, momentum_constraint,
    h_evolution_rhs, K_evolution_rhs,
    gauss_rhs, codazzi_rhs,
    to_latex,
)


def main() -> None:
    print("=" * 70)
    print("Backend 1.b: ADM 3+1 decomposition demo")
    print("=" * 70)

    st = IndexSpace("st", dim=4, indices="μνρσ", metric="g")
    sp = IndexSpace("sp", dim=3, indices="ijklmn", metric="h")
    print(f"  4D spacetime st  : dim={st.dim}, metric='{st.metric}'")
    print(f"  3D spatial sp    : dim={sp.dim}, metric='{sp.metric}'")

    adm = ADMSetup(st, sp)
    print()
    print("  Leaf builders:")
    print(f"    lapse N           : {to_latex(adm.lapse())}")
    print(f"    shift N^i         : {to_latex(adm.shift())}")
    print(f"    shift N_i         : {to_latex(adm.shift('i', 'lower'))}")
    print(f"    spatial h_{{ij}}    : {to_latex(adm.spatial_metric_lower())}")
    print(f"    spatial h^{{ij}}    : {to_latex(adm.spatial_metric_upper())}")
    print(f"    extrinsic K_{{ij}}  : {to_latex(adm.extrinsic_curvature())}")

    # ─── K_{ij} definition ────────────────────────────────────
    h_lo = adm.spatial_metric_lower()
    h_up = adm.spatial_metric_upper()
    conn3 = LeviCivitaConnection(h_lo, h_up, sp)

    K_def = extrinsic_curvature_definition(adm, conn3)
    print()
    print("  K_{ij} definition  (1/(2N)) (∂_t h_{ij} - D_i N_j - D_j N_i):")
    print(f"    {to_latex(K_def)}")
    print(f"    free   = {[i.name for i in K_def.free_indices]}")

    # ─── K trace ──────────────────────────────────────────────
    K_tr = K_trace_definition(adm)
    print()
    print(f"  K trace  K = h^{{ij}} K_{{ij}}:")
    print(f"    {to_latex(K_tr)}")
    print(f"    free   = {[i.name for i in K_tr.free_indices]}")

    # ─── Metric components ────────────────────────────────────
    print()
    print("  g_{μν} ADM components (lower):")
    for k, v in metric_lower_components(adm).items():
        free = [i.name for i in v.free_indices]
        print(f"    g_{{{k}}}  = {to_latex(v):<55s}    free={free}")

    print()
    print("  g^{μν} ADM components (upper):")
    for k, v in metric_upper_components(adm).items():
        free = [i.name for i in v.free_indices]
        print(f"    g^{{{k}}}  = {to_latex(v):<55s}    free={free}")

    # ─── Identities (Backend 1.b+ Gauss-Codazzi) ───────────────
    print()
    print("=" * 70)
    print("ADM identities (constraints, evolution, Gauss-Codazzi)")
    print("=" * 70)

    print()
    print("  Hamiltonian constraint  ℋ = R^(3) + K² - K_{ij}K^{ij}:")
    H = hamiltonian_constraint(adm)
    print(f"    {to_latex(H)}")
    print(f"    free   = {[i.name for i in H.free_indices]}")

    print()
    print("  Momentum constraint  ℋ_i = D_j K^j_i - D_i K:")
    Mc = momentum_constraint(adm, conn3)
    print(f"    {to_latex(Mc)}")
    print(f"    free   = {[i.name for i in Mc.free_indices]}")

    print()
    print("  h evolution  ∂_t h_{ij} = -2N K_{ij} + D_i N_j + D_j N_i:")
    he = h_evolution_rhs(adm, conn3)
    print(f"    {to_latex(he)}")
    print(f"    free   = {[i.name for i in he.free_indices]}")

    print()
    print("  K evolution  ∂_t K_{ij} = -D_i D_j N + N(R^(3)_{ij} + K K_{ij} - 2 K_ik K^k_j):")
    Ke = K_evolution_rhs(adm, conn3)
    print(f"    {to_latex(Ke)}")
    print(f"    free   = {[i.name for i in Ke.free_indices]}")

    print()
    print("  Gauss RHS  R^(3)_{ijkl} + K_{ik} K_{jl} - K_{il} K_{jk}:")
    Gr = gauss_rhs(adm)
    print(f"    {to_latex(Gr)}")
    print(f"    free   = {[i.name for i in Gr.free_indices]}")

    print()
    print("  Codazzi RHS  D_l K_{jk} - D_k K_{jl}:")
    Cr = codazzi_rhs(adm, conn3)
    print(f"    {to_latex(Cr)}")
    print(f"    free   = {[i.name for i in Cr.free_indices]}")

    # ─── Lie derivative + shift advection ───────────────────────
    print()
    print("=" * 70)
    print("Lie derivative + shift advection (B1.b 잔여)")
    print("=" * 70)

    # L_X V^a (vector)
    X = Tensor("X", [adm.st.upper("μ")])
    V = Tensor("V", [adm.st.upper("a")])
    Lie_V = LieDeriv(X, V)
    print()
    print("  L_X V^a (vector):")
    print(f"    compact   : {to_latex(Lie_V)}")
    print(f"    expanded  : {to_latex(expand_lie_deriv(Lie_V))}")

    # L_N K_ij (shift on K)
    N_vec = adm.shift("p", "upper")
    K_for_lie = adm.extrinsic_curvature("i", "j")
    Lie_K = LieDeriv(N_vec, K_for_lie)
    print()
    print("  L_N K_{ij} (shift on K):")
    print(f"    compact   : {to_latex(Lie_K)}")
    print(f"    expanded  : {to_latex(expand_lie_deriv(Lie_K))}")

    # K_evolution with shift advection
    print()
    print("  K_evolution (with shift advection):")
    Ke_full = K_evolution_rhs(adm, conn3, include_shift_advection=True)
    print(f"    {to_latex(Ke_full)}")

    # ─── slice_decompose ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("4D ↔ ADM split (slice_decompose)")
    print("=" * 70)

    g_lo_4d = Tensor(
        "g", [adm.st.lower("μ"), adm.st.lower("ν")],
        symmetric_pairs=[(0, 1)],
    )
    print()
    print("  slice_decompose(g_{μν}) — rank-2 4D metric:")
    for k, v in slice_decompose(g_lo_4d, adm.sp).items():
        free = [i.name for i in v.free_indices]
        print(f"    {k}: {to_latex(v):<20s}    free={free}")

    V_4d = Tensor("T", [adm.st.upper("μ")])
    print()
    print("  slice_decompose(T^μ) — rank-1 4D vector:")
    for k, v in slice_decompose(V_4d, adm.sp).items():
        free = [i.name for i in v.free_indices]
        print(f"    {k}: {to_latex(v):<20s}    free={free}")

    print()
    print("=" * 70)
    print("done.")


if __name__ == "__main__":
    main()
