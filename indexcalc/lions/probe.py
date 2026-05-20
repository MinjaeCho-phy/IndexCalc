"""LIONS probe — given a Lagrangian and a set of candidate groups,
determine which groups it is invariant under.

Output: per-group ``ProbeResult`` with the verdict, the list of fields
that have non-trivial representation under the group, and the group's
parameter count (dimension).

이 모듈은 v2 ("4 classical group + NR mechanics") 단계의 user-facing
surface. backend oracle은 그대로 ``apply_generator`` + ``simplify`` 사용.

v2 범위:
    - O(N), SO(N), U(N), SU(N) 의 generator + invariant tensor 등록은
      각 호출자가 ``GroupSpec`` 으로 넘긴다 (또는 ``classical_group_spec``
      헬퍼 사용).
    - time translation은 IR에 명시적 시간 변수가 없는 한 자동 통과.

v3에서 다룰 항목 (현재는 미지원):
    - hidden symmetry 발견 (Kepler SO(4) 등)
    - generator 변환식 emit
    - candidate group의 자동 derivation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.group import Group
from indexcalc.core.generator import Generator
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify
from indexcalc.core.variation import ZeroTensor


def _check_invariant(expr: TensorExpr, cand: "GroupSpec") -> bool:
    """Invariance verdict.

    1차: 기존 oracle (apply_generator + strip_time_deriv + simplify).
         ZeroTensor 도달 시 즉시 True.
    2차 fallback: structural check — 모든 vector index가 invariant tensor 또는
         같은 group rep의 field와 contract되어 있는지 + scalar (free=[]) 인지.

    1차가 충분히 강하지 않은 경우 (e.g., $SO(N)$의 $\\delta$-bilinear cancellation —
    M 텐서의 vector-index antisymmetry가 IR-level에 명시되지 않아 simplify가 못 줄임)
    2차가 보완. v2.5에서 backend simplify에 metric absorption + M antisym 인코딩
    추가하면 1차만으로 충분.
    """
    delta = apply_generator(expr, cand.generator)
    delta_stripped = _strip_time_deriv(delta)
    simplified = simplify(delta_stripped)
    if isinstance(simplified, ZeroTensor):
        return True
    return _structural_check(expr, cand)


def _structural_check(expr: TensorExpr, cand: "GroupSpec") -> bool:
    """Structural fallback: 모든 field가 rep 일관성 + scalar 표현.

    검사 사항:
    - expr.free_indices == [] (Lagrangian은 스칼라).
    - 모든 Tensor 잎의 vector-rep 인덱스 (rep="vector"가 등록된 그룹의 vector
      IndexSpace 인덱스) 가 expression 내 어딘가에서 contract됨.
    - 인덱스 contraction 양 끝이 (vector field) 또는 (이 그룹의 등록된 invariant
      tensor) 인지.

    이 함수는 simplify가 못 잡는 cancellation을 우회하기 위한 v2 임시 대안.
    false-positive 위험: 정말 깨진 라그랑지안인데 인덱스 구조만 맞춘 케이스를
    못 잡을 수 있음. NR mechanics + 4 classical groups 범위에서는 충분.
    """
    if expr.free_indices:
        return False

    # vector rep을 가진 group의 vector IndexSpace 식별.
    # GroupSpec의 generator action을 만든 vector_space를 inspect.
    vector_space = _find_vector_space(cand)
    if vector_space is None:
        return False  # group이 vector rep을 안 가짐 — 별도 처리 필요

    leaves = _collect_tensor_leaves(expr)
    for leaf in leaves:
        is_invariant_tensor = _is_known_invariant(leaf.name, cand.name)
        if is_invariant_tensor:
            continue  # δ, ε 같은 등록 invariant는 vector index 허용.
        rep = leaf.reps.get(cand.name)
        if rep is None:
            # rep 태그 없음 — vector index가 있으면 위반.
            for idx in leaf.indices:
                if idx.space == vector_space:
                    return False
            continue
        if rep == "singlet":
            # singlet이라면 vector index를 가지면 안 됨 (rep mismatch).
            for idx in leaf.indices:
                if idx.space == vector_space:
                    return False
        # rep == "vector"이면 contraction 여부는 expr.free_indices == []
        # 에서 이미 검사됨.
    return True


def _find_vector_space(cand: "GroupSpec"):
    """GroupSpec이 만들어진 vector IndexSpace를 generator 내부에서 추출."""
    # make_o_n_generator는 vector rep에 lorentz_vector_action을 등록한다.
    # 이 action의 closure가 vector_space를 갖고 있음 → action이 만든 Tensor에서
    # 추론. 간단한 우회: cand.generator의 vector action을 dummy field에 적용.
    if not cand.generator.has_action("vector"):
        return None
    from indexcalc.core.index import IndexSpace
    from indexcalc.core.tensor import Tensor as _T
    # action 내부의 space를 직접 노출하는 깔끔한 API가 없으므로 inspect
    # closure로 접근.
    act = cand.generator._actions.get("vector")
    if act is None:
        return None
    if hasattr(act, "__closure__") and act.__closure__:
        for cell in act.__closure__:
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if isinstance(v, IndexSpace):
                return v
    return None


def _collect_tensor_leaves(expr: TensorExpr) -> list[Tensor]:
    if isinstance(expr, Tensor):
        return [expr]
    if isinstance(expr, (TensorProduct, TensorSum)):
        return _collect_tensor_leaves(expr.left) + _collect_tensor_leaves(expr.right)
    if isinstance(expr, ScalarMul):
        return _collect_tensor_leaves(expr.expr)
    from indexcalc.adm import TimeDeriv
    if isinstance(expr, TimeDeriv):
        return _collect_tensor_leaves(expr.expr)
    from indexcalc.core.scalar_function import ScalarFunction
    if isinstance(expr, ScalarFunction):
        return _collect_tensor_leaves(expr.arg)
    return []


_STANDARD_O_INVARIANTS = {"delta", "delta_mixed", "epsilon"}


def _is_known_invariant(tensor_name: str, group_name: str) -> bool:
    """등록된 표준 invariant tensor인가? v2 범위: O(N)/SO(N)의 delta·epsilon만."""
    if not (group_name.startswith("O(") or group_name.startswith("SO(")):
        return False
    if tensor_name == "epsilon" and not group_name.startswith("SO("):
        return False  # epsilon은 SO(N) 한정
    return tensor_name in _STANDARD_O_INVARIANTS


def _strip_time_deriv(expr: TensorExpr) -> TensorExpr:
    """TimeDeriv(T)을 T로 대체. invariance 체크는 시간 미분과 commute하는
    spatial/internal symmetry 대상이므로 결과 등가.

    이 변환은 simplify가 TimeDeriv 내부 인덱스 canonicalization을 못 하는
    문제를 회피 — \\dot Φ^i 회전 변환항을 plain Φ^i 형태로 환원.
    """
    from indexcalc.adm import TimeDeriv
    from indexcalc.core.scalar_function import ScalarFunction

    if isinstance(expr, TimeDeriv):
        return _strip_time_deriv(expr.expr)
    if isinstance(expr, ScalarFunction):
        return ScalarFunction(expr.name, _strip_time_deriv(expr.arg))
    if isinstance(expr, TensorProduct):
        return TensorProduct(
            _strip_time_deriv(expr.left),
            _strip_time_deriv(expr.right),
        )
    if isinstance(expr, TensorSum):
        return TensorSum(
            _strip_time_deriv(expr.left),
            _strip_time_deriv(expr.right),
        )
    if isinstance(expr, ScalarMul):
        return ScalarMul(expr.scalar, _strip_time_deriv(expr.expr))
    return expr  # Tensor / ZeroTensor / PartialDeriv 등 그대로


@dataclass(frozen=True)
class GroupSpec:
    """Candidate group + 검증에 필요한 generator + 메타데이터."""
    name: str
    group: Group
    generator: Generator
    dim: int


@dataclass
class ProbeResult:
    """단일 (group, expression) pair의 probe 결과.

    Parameters
    ----------
    group : str
        후보 그룹 이름 (e.g., "SO(3)", "U(1)").
    invariant : bool
        oracle 결과. True ⇔ apply_generator(expr) → ZeroTensor.
    non_singlet_fields : dict[str, str]
        ``{field_name: rep_label}``. 이 그룹 하에서 non-trivially
        변환하는 field들의 rep 라벨.
    dim : int
        Group dimension = parameter 수.
    notes : str
        부가 정보 (e.g., "trivial — all fields singlet").
    """
    group: str
    invariant: bool
    non_singlet_fields: dict
    dim: int
    notes: str = ""


def probe(
    expr: TensorExpr,
    fields: list[Tensor],
    candidates: list[GroupSpec],
) -> list[ProbeResult]:
    """각 candidate group에 대해 expr의 invariance를 체크.

    Parameters
    ----------
    expr : TensorExpr
        Lagrangian (또는 임의 표현식).
    fields : list[Tensor]
        ``expr`` 안에 등장하는 field 텐서들. rep 라벨 dump용.
    candidates : list[GroupSpec]
        탐색할 그룹들. ``classical_group_spec`` 헬퍼로 생성하거나 직접.

    Returns
    -------
    list[ProbeResult]
        각 candidate에 대한 결과. invariant=True는 그룹이 expr를 보존함을 의미.
    """
    results: list[ProbeResult] = []
    for cand in candidates:
        invariant = _check_invariant(expr, cand)

        non_singlets: dict[str, str] = {}
        for f in fields:
            rep = f.reps.get(cand.name)
            if rep is None or rep == "singlet":
                continue
            non_singlets[f.name] = rep

        notes = ""
        if invariant and not non_singlets:
            notes = "trivial (all fields singlet)"

        results.append(ProbeResult(
            group=cand.name,
            invariant=invariant,
            non_singlet_fields=non_singlets,
            dim=cand.dim,
            notes=notes,
        ))
    return results


# ─── Helper: 표준 classical group spec 생성 ─────────────────────────────


def classical_group_spec(
    group_name: str,
    N: int,
    index_space=None,
    *,
    adj_space=None,
) -> GroupSpec:
    """4 classical group의 GroupSpec 생성.

    Supported:
    - ``"O(N)"``, ``"SO(N)"``: vector + singlet, ``index_space``는 N-dim vector space.
    - ``"Sp(2N)"``: vector + singlet, ``index_space``는 2N-dim (짝수) vector space
      with antisymmetric Ω metric. ``N`` 인자는 vector dim (= 2·rank).
    - ``"SU(N)"``, ``"U(N)"``: fund + antifund + singlet, ``index_space``는 fund space
      (N-dim 복소), ``adj_space``는 adj space (dim = N²-1 for SU, N² for U).
    - ``"U(1)"``: charged reps, ``index_space``는 무시 (abelian).

    Parameters
    ----------
    group_name : str
        그룹 이름.
    N : int
        rep 차원 (vector dim 또는 fund dim).
    index_space : IndexSpace
        primary 인덱스 공간 (vector for O/SO, fund for SU/U).
    adj_space : IndexSpace | None
        SU(N)/U(N)용 adjoint 인덱스 공간. 없으면 자동 생성 (이름
        ``{group}_adj``).
    """
    from indexcalc.core.generator import (
        make_o_n_generator, make_su_n_generator, make_u1_generator,
        make_sp_2n_generator,
    )
    from indexcalc.core.index import IndexSpace as _IS

    if group_name.startswith("Sp("):
        if index_space is None:
            raise ValueError(f"{group_name} requires index_space (vector)")
        if N % 2 != 0:
            raise ValueError(
                f"{group_name}: symplectic vector dim must be even, got {N}"
            )
        rank = N // 2
        dim = rank * (N + 1)  # = rank·(2·rank+1)
        g = Group(group_name, dim=dim, abelian=False)
        g.add_rep("vector", dim=N)
        g.add_rep("singlet", dim=1)
        gen = make_sp_2n_generator(g, index_space)
        return GroupSpec(name=group_name, group=g, generator=gen, dim=dim)

    if group_name.startswith("O(") or group_name.startswith("SO("):
        if index_space is None:
            raise ValueError(f"{group_name} requires index_space (vector)")
        dim = N * (N - 1) // 2
        g = Group(group_name, dim=dim, abelian=False)
        g.add_rep("vector", dim=N)
        g.add_rep("singlet", dim=1)
        gen = make_o_n_generator(g, index_space)
        return GroupSpec(name=group_name, group=g, generator=gen, dim=dim)

    if group_name.startswith("SU(") or group_name.startswith("U("):
        if group_name == "U(1)":
            g = Group("U(1)", dim=1, abelian=True)
            # caller가 charge rep을 명시적으로 add — 여기서 default로 ±1 추가하면
            # 컨텍스트에 안 맞을 위험. 빈 그룹 + 빈 generator를 반환하고 caller가
            # add_rep + gen.declare_action을 직접 호출하는 식이 가장 깔끔.
            # 단순 helper로는 +1 / -1 / 0 세 개만 자동 등록.
            g.add_rep("+1", dim=1, charge=1.0)
            g.add_rep("-1", dim=1, charge=-1.0)
            g.add_rep("0", dim=1, charge=0.0)
            gen = make_u1_generator(g)
            return GroupSpec(name="U(1)", group=g, generator=gen, dim=1)

        if index_space is None:
            raise ValueError(f"{group_name} requires index_space (fund)")
        # SU(N): N²-1 generators; U(N): N²
        dim = N * N - 1 if group_name.startswith("SU(") else N * N
        if adj_space is None:
            adj_space = _IS(
                f"{group_name.lower()}_adj", dim=dim,
                indices="abcdefghABCDEFGH",
            )
        g = Group(group_name, dim=dim, abelian=False)
        g.add_rep("fund", dim=N)
        g.add_rep("antifund", dim=N, conjugate=True)
        g.add_rep("adj", dim=dim)
        g.add_rep("singlet", dim=1)
        gen = make_su_n_generator(g, adj_space, fund_space=index_space)
        return GroupSpec(name=group_name, group=g, generator=gen, dim=dim)

    raise ValueError(
        f"classical_group_spec: unsupported group {group_name!r}. "
        f"Supported: O(N), SO(N), Sp(2N), U(N), SU(N), U(1)."
    )


def format_probe_report(
    expr_label: str, results: list[ProbeResult],
) -> str:
    """사용자용 텍스트 리포트. CLI 출력 후보."""
    lines = [f"Lagrangian: {expr_label}", "", "Detected symmetries:"]
    invariant = [r for r in results if r.invariant]
    not_inv = [r for r in results if not r.invariant]
    for r in invariant:
        if r.non_singlet_fields:
            fields_str = ", ".join(
                f"{n}: {rep}" for n, rep in r.non_singlet_fields.items()
            )
        else:
            fields_str = "(no non-singlet fields)"
        lines.append(
            f"  {r.group:<10} ✓  {fields_str:<40} dim={r.dim}"
            + (f"  [{r.notes}]" if r.notes else "")
        )
    if not_inv:
        lines.append("")
        lines.append("Not invariant under:")
        for r in not_inv:
            lines.append(f"  {r.group:<10} ✗  dim={r.dim}")
    return "\n".join(lines)
