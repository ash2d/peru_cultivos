"""One YAML loader for every config in ``config/``, with ``extends:``.

Why this exists: the configs were a **copy-forward chain**. ``perennial_allperu.yaml``
was ``perennial.yaml`` pasted whole plus ~200 sierra/selva tokens, and
``perennial_cenagro.yaml`` was *that* pasted whole plus the census vocabulary — 559 lines
each, of which the part that mattered was a few dozen. Two headers deep, both said
"Original header follows". The split configs were the same story: six files differing from
one another by one or two keys.

That is not just verbose, it is the failure mode this project keeps writing down. A
derived lexicon is only safe if it is **additive** — if it changed an existing token's
group, a measured "change in the land" would partly be a change of definition
(``RESULTS.md`` §8.5). When the derived file is a full copy, nothing enforces that and
nothing shows it; the reader has to diff 559 lines to find out.

So:

``extends: <file>``
    Load that file first (relative to this one's directory, recursively), then let this
    file's own top-level keys **replace** the parent's. Use it for a config that is a
    genuine variant — a different label space, a different buffer width.

``add: {key: [...]}``
    **Append** to the parent's list under ``key``, in order, skipping exact duplicates.
    This is the additive-only discipline, written down where it is used instead of
    asserted in a comment. A key that is not already a list in the parent raises, so a
    typo becomes an error rather than a silently-ignored new group.

``drop: [key, ...]``
    Delete inherited top-level keys. Needed only where a variant genuinely renames a
    group (``perennial_binary.yaml`` has ``non_perennial`` where the base has ``annual``),
    and it is explicit so that a reader can see what left.

⚠️ **The loader does not check that a token stays in one group** — it does not need to.
``perennial.labels3.build_resolver`` already raises when a token resolves twice, and it
sees the *merged* config, so an ``add:`` that collides with an inherited assignment fails
there. That check is the one of record; this module must not duplicate it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_MERGE_KEYS = ("extends", "add", "drop")


def load_yaml_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load ``path``, resolving ``extends`` / ``add`` / ``drop``.

    Returns a plain dict with the three merge keys removed, so every existing consumer
    sees exactly what it saw when the file was a full copy.
    """
    path = Path(path).resolve()
    if path in _seen:
        chain = " -> ".join(p.name for p in (*_seen, path))
        raise ValueError(f"circular `extends` in config chain: {chain}")

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"{path} does not parse to a mapping")

    parent_name = cfg.pop("extends", None)
    add = cfg.pop("add", None) or {}
    drop = cfg.pop("drop", None) or []

    if parent_name is None:
        if add or drop:
            raise ValueError(f"{path.name}: `add`/`drop` need an `extends:` to act on")
        return cfg

    merged = load_yaml_config(path.parent / parent_name, (*_seen, path))

    for key, extra in add.items():
        base = merged.get(key)
        if not isinstance(base, list):
            raise KeyError(
                f"{path.name}: `add: {key}` — the parent "
                f"({parent_name}) has no list under {key!r}. "
                f"Lists it does have: {sorted(k for k, v in merged.items() if isinstance(v, list))}"
            )
        if not isinstance(extra, list):
            raise TypeError(f"{path.name}: `add: {key}` must be a list, got {type(extra).__name__}")
        merged[key] = base + [x for x in extra if x not in base]

    for key in drop:
        if key not in merged:
            raise KeyError(f"{path.name}: `drop: {key}` — the parent has no such key")
        del merged[key]

    merged.update(cfg)
    return merged
