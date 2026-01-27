from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MemoryUnit:
    """A single atomic memory item stored for retrieval."""

    id: int
    text: str
    unit_type: str = "fact"  # e.g., preference, decision, commitment, summary, ...
    entities: List[str] = field(default_factory=list)
    confidence: float = 0.8
    created_at: Optional[str] = None
    source_refs: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True


@dataclass
class UserState:
    """Small durable user profile / tasks / entities registry.

    Keep this small; it is injected into the interaction agent prompt.
    """

    profile: List[str] = field(default_factory=list)
    open_loops: List[Dict[str, Any]] = field(default_factory=list)
    commitments: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)  # [{canonical,type,aliases[]}]
    updated_at: Optional[str] = None

    @staticmethod
    def empty() -> "UserState":
        return UserState(updated_at=datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "open_loops": self.open_loops,
            "commitments": self.commitments,
            "entities": self.entities,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserState":
        return cls(
            profile=[s for s in (data.get("profile") or []) if isinstance(s, str)],
            open_loops=[d for d in (data.get("open_loops") or []) if isinstance(d, dict)],
            commitments=[d for d in (data.get("commitments") or []) if isinstance(d, dict)],
            entities=[d for d in (data.get("entities") or []) if isinstance(d, dict)],
            updated_at=data.get("updated_at") if isinstance(data.get("updated_at"), str) else None,
        )
