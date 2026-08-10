from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalanche.judges.criteria import CriteriaJudge

__all__ = ["CriteriaJudge"]


def __getattr__(name: str) -> Any:
    if name == "CriteriaJudge":
        from evalanche.judges.criteria import CriteriaJudge

        return CriteriaJudge
    raise AttributeError(name)
