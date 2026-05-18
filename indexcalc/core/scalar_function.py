"""ScalarFunction: invariant scalar의 임의 함수 f(I).

비다항 lagrangian 항 (e.g., 1/sqrt(\\delta_{kl}\\Phi^k\\Phi^l), V(|\\Phi|^2),
e^{-\\phi^2/2})을 그룹-singlet으로 wrapping. backend는 인수 I가 free index
없는 스칼라이고, 후보 그룹 G의 invariant tensor만으로 구성됨을 확인하면
``f(I)``가 G-singlet임을 자동 도출.

f 자체는 opaque (Leibniz 전개 없음). variation/EOM 도출은 v3 이후.
"""

from __future__ import annotations
from indexcalc.core.index import Index
from indexcalc.core.tensor import TensorExpr


class ScalarFunction(TensorExpr):
    """f(I) — invariant scalar I 위에서 정의된 임의 함수.

    Parameters
    ----------
    name : str
        함수 라벨 (e.g., "inv_sqrt", "exp", "V").
    arg : TensorExpr
        인수. 반드시 free_indices == [] (스칼라).

    Raises
    ------
    ValueError
        ``arg.free_indices``가 비어있지 않으면.
    """

    def __init__(self, name: str, arg: TensorExpr):
        if arg.free_indices:
            raise ValueError(
                f"ScalarFunction argument must be a scalar (no free indices), "
                f"got free_indices={arg.free_indices}"
            )
        self.name = name
        self.arg = arg

    @property
    def free_indices(self) -> list[Index]:
        return []

    def __repr__(self) -> str:
        return f"{self.name}({self.arg})"
