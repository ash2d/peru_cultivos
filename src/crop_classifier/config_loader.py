"""One YAML loader for every config in ``config/``, with ``extends:``.

Why this exists: the configs were a **copy-forward chain**. ``perennial_allperu.yaml`` was
``perennial.yaml`` pasted whole plus ~200 sierra/selva tokens; ``perennial_cenagro.yaml``
was *that* plus the census vocabulary — 559 lines each, of which a few dozen mattered. The
split configs were the same: six files differing by one or two keys.

That is the failure mode this project keeps writing down. A derived lexicon is safe only if
**additive** — changing an existing token's group makes a measured "change in the land"
partly a change of definition (``RESULTS.md`` §8.5). A full-copy derived file neither
enforces nor shows that; the reader must diff 559 lines.

So:

``extends: <file>``
    Load that file first (relative to this one's directory, recursively), then let this
    file's own top-level keys **replace** the parent's. Use it for a genuine variant — a
    different label space, a different buffer width.

``add: {key: [...]}``
    **Append** to the parent's list under ``key``, in order, skipping exact duplicates — the
    additive-only discipline, enforced where it is used. A key that is not already a list in
    the parent raises, so a typo is an error not a silent new group.

``drop: [key, ...]``
    Delete inherited top-level keys. Only where a variant renames a group
    (``perennial_binary.yaml`` has ``non_perennial`` where the base has ``annual``);
    explicit so a reader sees what left.

⚠️ **The loader does not check that a token stays in one group** — it need not.
``perennial.labels3.build_resolver`` already raises when a token resolves twice, on the
*merged* config, so a colliding ``add:`` fails there. That check is the one of record; do
not duplicate it here.
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
