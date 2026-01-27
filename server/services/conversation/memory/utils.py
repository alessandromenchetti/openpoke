from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set


_WS_RE = re.compile(r"[ \t\r\n]+")
_PUNC_RE = re.compile(r"[^a-z0-9@._ \-]+")


def normalize_entity(text: str) -> str:
    """Normalize an entity string for matching.

    - lowercases
    - collapses whitespace
    - strips most punctuation (keeps @ . _ -)
    """
    lowered = (text or "").strip().lower()
    lowered = _PUNC_RE.sub(" ", lowered)
    lowered = _WS_RE.sub(" ", lowered).strip()
    return lowered


def entity_aliases(entity: str) -> Set[str]:
    """Return a set of alias strings for a single entity mention."""
    norm = normalize_entity(entity)
    if not norm:
        return set()

    aliases: Set[str] = {norm}

    # For full names, include first/last tokens as weaker aliases.
    tokens = [t for t in norm.split(" ") if t]
    if len(tokens) >= 2:
        aliases.add(tokens[0])
        aliases.add(tokens[-1])
        aliases.add(" ".join(tokens[:2]))

    # Email / handle: include local-part.
    if "@" in norm:
        local = norm.split("@", 1)[0].strip()
        if local:
            aliases.add(local)

    return aliases


def build_alias_index(state_entities: List[Dict[str, object]]) -> Dict[str, List[str]]:
    """Build alias -> [canonical] index from state entities."""
    alias_to_canon: Dict[str, List[str]] = {}
    for ent in state_entities:
        canonical = ent.get("canonical")
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        canon_norm = normalize_entity(canonical)
        aliases = ent.get("aliases")
        if isinstance(aliases, list):
            alias_list = [a for a in aliases if isinstance(a, str) and a.strip()]
        else:
            alias_list = []

        # Always include canonical itself as an alias.
        for a in set(alias_list + [canonical]):
            for alias in entity_aliases(a):
                alias_to_canon.setdefault(alias, [])
                if canon_norm not in alias_to_canon[alias]:
                    alias_to_canon[alias].append(canon_norm)
    return alias_to_canon


def is_ambiguous_alias(alias: str, alias_index: Dict[str, List[str]]) -> bool:
    vals = alias_index.get(alias) or []
    return len(vals) > 1


def entity_match(
    query_entities: Iterable[str],
    unit_entities: Iterable[str],
    alias_index: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """Heuristic entity match supporting first-name vs full-name.

    We treat single-token matches as "weak" and ignore them if ambiguous
    (e.g., multiple people named "john" in the entity registry).
    """
    q_aliases: Set[str] = set()
    for e in query_entities:
        q_aliases |= entity_aliases(e)

    u_aliases: Set[str] = set()
    for e in unit_entities:
        u_aliases |= entity_aliases(e)

    if not q_aliases or not u_aliases:
        return False

    common = q_aliases & u_aliases
    if not common:
        return False

    if alias_index:
        # If the only overlap is an ambiguous single-token alias, reject.
        for a in list(common):
            if " " not in a and is_ambiguous_alias(a, alias_index):
                common.discard(a)
        if not common:
            return False

    return True


def ensure_entity_in_registry(
    state_entities: List[Dict[str, object]],
    mention: str,
    entity_type: str = "unknown",
) -> None:
    """Upsert an entity mention into the state registry."""
    norm = normalize_entity(mention)
    if not norm:
        return

    # Find canonical match.
    for ent in state_entities:
        canonical = ent.get("canonical")
        if isinstance(canonical, str) and normalize_entity(canonical) == norm:
            # add aliases if missing
            aliases = ent.get("aliases")
            if not isinstance(aliases, list):
                ent["aliases"] = [mention]
            else:
                if mention not in aliases:
                    aliases.append(mention)
            return

    # Insert new entity.
    state_entities.append(
        {
            "canonical": mention.strip(),
            "type": entity_type,
            "aliases": [mention.strip()],
        }
    )