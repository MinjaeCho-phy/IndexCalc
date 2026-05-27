"""
TensorExpr: 텐서 표현식의 symbolic 표현을 위한 트리 구조.

모든 텐서 표현식은 TensorExpr의 서브클래스이다:
  - Tensor:        잎 노드. 이름과 인덱스를 가진 단일 텐서 (e.g., T^μ_ν).
  - TensorProduct: 두 표현식의 곱. 반복 인덱스를 자동으로 contraction으로 인식.
  - TensorSum:     두 표현식의 합. free index 구조가 일치해야 한다.
  - ScalarMul:     스칼라 × 텐서 표현식.

Python 연산자 *, +, -를 오버로딩하여 자연스럽게 표현식을 조합할 수 있다.
"""

from __future__ import annotations
from indexcalc.core.index import Index, IndexSpace


def _resolve_einstein_pairs(indices: list[Index]) -> list[Index]:
    """Einstein convention 자동 contraction: 한 노드 안에서 같은 이름 + 같은
    IndexSpace + 반대 position인 인덱스가 정확히 두 번 등장하면 그 쌍을 dummy로
    제거한다.

    ``Tensor`` (e.g., $\\Gamma^\\rho{}_{\\rho\\lambda}$ — slot 0 ↔ slot 1)와
    ``PartialDeriv``/``CovariantDeriv`` (deriv_index ↔ inner의 free index)에서
    공통으로 사용. 같은 이름이 1번이면 free, 같은 위치로 2번이면 그대로 둠
    (Einstein 위반은 ``validate_einstein``에서 별도로 잡는다).
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for i, idx in enumerate(indices):
        key = (idx.name, idx.space.name)
        groups.setdefault(key, []).append(i)

    to_remove: set[int] = set()
    for positions in groups.values():
        if len(positions) != 2:
            continue
        i1, i2 = positions
        if indices[i1].position != indices[i2].position:
            to_remove.add(i1)
            to_remove.add(i2)

    return [idx for i, idx in enumerate(indices) if i not in to_remove]


class TensorExpr:
    """모든 텐서 표현식의 기반 클래스.

    직접 인스턴스화하지 않고, 서브클래스(Tensor, TensorProduct 등)를 사용한다.
    Python 연산자를 통해 표현식 트리를 구성한다:
      - A * B  →  TensorProduct(A, B)   (반복 인덱스 자동 contraction)
      - A + B  →  TensorSum(A, B)
      - A - B  →  TensorSum(A, -B)
      - -A     →  ScalarMul(-1, A)
      - 3 * A  →  ScalarMul(3, A)
    """

    @property
    def free_indices(self) -> list[Index]:
        """이 표현식의 free index(축약되지 않고 남은 인덱스) 목록."""
        raise NotImplementedError

    @property
    def rank(self) -> tuple[int, int]:
        """(upper 개수, lower 개수) 형태의 텐서 rank."""
        free = self.free_indices
        n_upper = sum(1 for i in free if i.position == "upper")
        n_lower = sum(1 for i in free if i.position == "lower")
        return (n_upper, n_lower)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ScalarMul(other, self)
        if isinstance(other, TensorExpr):
            return TensorProduct(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return ScalarMul(other, self)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, TensorExpr):
            return TensorSum(self, other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, TensorExpr):
            return TensorSum(self, ScalarMul(-1, other))
        return NotImplemented

    def __neg__(self):
        return ScalarMul(-1, self)

    def to_latex(self) -> str:
        """이 표현식을 LaTeX 문자열로 변환한다 ($ 없이)."""
        from indexcalc.parse.display import to_latex
        return to_latex(self)

    def _repr_latex_(self) -> str:
        """Jupyter notebook에서 LaTeX 렌더링을 위한 메서드."""
        return f"${self.to_latex()}$"


class Tensor(TensorExpr):
    """이름과 인덱스를 가진 단일 텐서. 표현식 트리의 잎(leaf) 노드.

    Parameters
    ----------
    name : str
        텐서의 이름 (e.g., "T", "g", "R").
    indices : list[Index]
        텐서에 달린 인덱스들. 순서가 의미를 가진다.

    Examples
    --------
    >>> st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
    >>> mu, nu = st.upper("μ"), st.lower("ν")
    >>> T = Tensor("T", [mu, nu])
    >>> T
    T^μ_ν
    >>> T.rank
    (1, 1)
    """

    def __init__(
        self,
        name: str,
        indices: list[Index],
        antisymmetric_pairs: list[tuple[int, int]] | None = None,
        *,
        symmetric_pairs: list[tuple[int, int]] | None = None,
        traceless: list[tuple[int, int]] | None = None,
        transverse: list[int] | None = None,
        cometric_antisymmetric_pairs: list[tuple[int, int]] | None = None,
        reps: dict[str, str] | None = None,
        statistics: str = "bosonic",
        is_coordinate: bool = False,
    ):
        self.name = name
        self.indices = tuple(indices)
        # is_coordinate: 시공간 좌표 x^μ. ∂_ν x^μ = δ^μ_ν (Kronecker) — 일반
        # 비-field leaf의 ∂=0 과 다른 유일한 예외. noether._push_div 이 처리.
        # 좌표는 정확히 하나의 upper 인덱스만 (lower 좌표는 η_μν x^ν로 표기).
        self.is_coordinate = is_coordinate
        # antisymmetric_pairs: 서로 바꾸면 -1 배인 slot 쌍 목록. 예: B_μν = -B_νμ
        # → antisymmetric_pairs=[(0, 1)]
        self.antisymmetric_pairs: tuple[tuple[int, int], ...] = tuple(
            tuple(sorted(p)) for p in (antisymmetric_pairs or [])
        )
        # symmetric_pairs: 서로 바꿔도 그대로인 slot 쌍 목록. 예: g_μν = g_νμ
        self.symmetric_pairs: tuple[tuple[int, int], ...] = tuple(
            tuple(sorted(p)) for p in (symmetric_pairs or [])
        )
        # traceless: γ^ij T_ij = 0 이 성립하는 slot 쌍 목록.
        #   typically symmetric_pairs 안의 쌍과 일치.
        self.traceless: tuple[tuple[int, int], ...] = tuple(
            tuple(sorted(p)) for p in (traceless or [])
        )
        # transverse: ∂^i T_..i.. = 0 이 성립하는 slot 인덱스 목록.
        self.transverse: tuple[int, ...] = tuple(transverse or [])
        # cometric_antisymmetric_pairs: 두 slot이 **같은 position에 올 때만**
        # antisymmetric인 쌍 (metric raise/lower 후에 드러나는 antisymmetry).
        # 예: SO(N) vector-rep generator M^{ab,i}_j 는 mixed position에선 일반
        # 행렬이지만 i,j 둘 다 내리면 so(N)=반대칭. 생성 시점엔 조건부라 명시
        # 불가 → simplify의 promote_cometric_antisym 패스가 같은-position에 온
        # 쌍을 antisymmetric_pairs로 승격한다.
        self.cometric_antisymmetric_pairs: tuple[tuple[int, int], ...] = tuple(
            tuple(sorted(p)) for p in (cometric_antisymmetric_pairs or [])
        )

        # ── validation ──────────────────────────────────────
        n = len(self.indices)
        seen_pairs: set[tuple[int, int]] = set()
        for p in self.antisymmetric_pairs:
            if p[0] == p[1] or p[0] < 0 or p[1] >= n:
                raise ValueError(f"invalid antisymmetric_pair {p} for rank {n}")
            seen_pairs.add(p)
        for p in self.symmetric_pairs:
            if p[0] == p[1] or p[0] < 0 or p[1] >= n:
                raise ValueError(f"invalid symmetric_pair {p} for rank {n}")
            if p in seen_pairs:
                raise ValueError(
                    f"slot pair {p} is both symmetric and antisymmetric"
                )
            seen_pairs.add(p)
        for p in self.traceless:
            if p[0] == p[1] or p[0] < 0 or p[1] >= n:
                raise ValueError(f"invalid traceless pair {p} for rank {n}")
            # traceless slot 쌍의 두 인덱스는 같은 IndexSpace여야 함.
            if self.indices[p[0]].space != self.indices[p[1]].space:
                raise ValueError(
                    f"traceless pair {p} crosses different IndexSpaces"
                )
        for s in self.transverse:
            if s < 0 or s >= n:
                raise ValueError(f"invalid transverse slot {s} for rank {n}")
        for p in self.cometric_antisymmetric_pairs:
            if p[0] == p[1] or p[0] < 0 or p[1] >= n:
                raise ValueError(
                    f"invalid cometric_antisymmetric_pair {p} for rank {n}"
                )
            if self.indices[p[0]].space != self.indices[p[1]].space:
                raise ValueError(
                    f"cometric_antisymmetric_pair {p} crosses different IndexSpaces"
                )

        # reps: {group_name: rep_name}. 비어있으면 모든 그룹의 singlet으로 간주.
        self.reps: dict[str, str] = dict(reps) if reps else {}
        if statistics not in ("bosonic", "fermionic"):
            raise ValueError(
                f"statistics must be 'bosonic' or 'fermionic', got {statistics!r}"
            )
        self.statistics = statistics
        if self.is_coordinate and not (
            len(self.indices) == 1 and self.indices[0].position == "upper"
        ):
            raise ValueError(
                "coordinate tensor must carry exactly one upper index "
                f"(x^μ), got {self.indices}"
            )

    @property
    def free_indices(self) -> list[Index]:
        """Self-contracted (같은 이름·반대 위치) 쌍을 dummy로 제거한 free 리스트.

        예: $\\Gamma^\\rho{}_{\\rho\\lambda}$ (indices=[ρ↑, ρ↓, λ↓]) →
        free=[λ↓]. ``Tensor.indices``는 원본을 보존; display나 저장 용도.
        """
        return _resolve_einstein_pairs(list(self.indices))

    def __repr__(self) -> str:
        if not self.indices:
            return self.name
        parts = []
        for idx in self.indices:
            if idx.position == "upper":
                parts.append(f"^{idx.name}")
            else:
                parts.append(f"_{idx.name}")
        return self.name + "".join(parts)


class TensorProduct(TensorExpr):
    """두 텐서 표현식의 곱. Einstein convention에 따라 반복 인덱스를 자동 축약한다.

    같은 이름 + 같은 공간 + 반대 위치(upper↔lower)인 인덱스 쌍을 찾아
    contracted pair로 분류하고, 나머지를 free index로 남긴다.

    Parameters
    ----------
    left : TensorExpr
        곱의 왼쪽 인자.
    right : TensorExpr
        곱의 오른쪽 인자.

    Raises
    ------
    ValueError
        같은 인덱스가 3번 이상 나타나는 경우 (유효하지 않은 Einstein notation).

    Examples
    --------
    >>> st = IndexSpace("spacetime", dim=4, metric="g")
    >>> mu, nu, lam = st.upper("μ"), st.lower("ν"), st.upper("λ")
    >>> T = Tensor("T", [mu, st.lower("ν")])    # T^μ_ν
    >>> S = Tensor("S", [st.upper("ν"), lam])    # S^ν_λ  (오타 아님: upper ν)
    -- 하지만 contraction이 되려면 T의 _ν와 S의 ^ν이어야 함 --
    """

    def __init__(self, left: TensorExpr, right: TensorExpr):
        self.left = left
        self.right = right

        left_indices = left.free_indices
        right_indices = right.free_indices

        # 전체 인덱스에서 같은 이름이 3번 이상 나오면 에러
        all_names = [i.name for i in left_indices + right_indices]
        for name in set(all_names):
            if all_names.count(name) > 2:
                raise ValueError(
                    f"Index '{name}' appears {all_names.count(name)} times. "
                    f"Einstein notation allows at most 2."
                )

        # Contraction 쌍 찾기
        self._contracted: list[tuple[Index, Index]] = []
        self._free: list[Index] = []

        right_remaining = list(right_indices)

        for li in left_indices:
            matched = False
            for j, ri in enumerate(right_remaining):
                if li.contracts_with(ri):
                    self._contracted.append((li, ri))
                    right_remaining.pop(j)
                    matched = True
                    break
            if not matched:
                self._free.append(li)

        self._free.extend(right_remaining)

    @property
    def free_indices(self) -> list[Index]:
        return list(self._free)

    @property
    def contracted_pairs(self) -> list[tuple[Index, Index]]:
        """축약된 인덱스 쌍들의 목록."""
        return list(self._contracted)

    def __repr__(self) -> str:
        contracted_info = ""
        if self._contracted:
            pairs = ", ".join(f"{a.name}" for a, _ in self._contracted)
            contracted_info = f"  [contracted: {pairs}]"
        return f"({self.left} * {self.right}){contracted_info}"


class TensorSum(TensorExpr):
    """두 텐서 표현식의 합.

    더해지는 두 표현식은 같은 수의 free index를 가져야 한다.
    (Phase 4에서 index 구조 일치 검증을 더 엄격하게 할 예정.)

    Parameters
    ----------
    left : TensorExpr
        합의 왼쪽 항.
    right : TensorExpr
        합의 오른쪽 항.

    Raises
    ------
    ValueError
        Free index 개수가 다른 경우.
    """

    def __init__(self, left: TensorExpr, right: TensorExpr):
        left_free = left.free_indices
        right_free = right.free_indices

        if len(left_free) != len(right_free):
            raise ValueError(
                f"Cannot add tensors with different free index count: "
                f"{len(left_free)} vs {len(right_free)}"
            )

        self.left = left
        self.right = right

    @property
    def free_indices(self) -> list[Index]:
        return self.left.free_indices

    def __repr__(self) -> str:
        return f"({self.left} + {self.right})"


class ScalarMul(TensorExpr):
    """스칼라와 텐서 표현식의 곱.

    Parameters
    ----------
    scalar : int or float
        스칼라 값.
    expr : TensorExpr
        텐서 표현식.
    """

    def __init__(self, scalar, expr: TensorExpr):
        self.scalar = scalar
        self.expr = expr

    @property
    def free_indices(self) -> list[Index]:
        return self.expr.free_indices

    def __repr__(self) -> str:
        if self.scalar == -1:
            return f"(-{self.expr})"
        return f"{self.scalar} * {self.expr}"
