"""Shared knowledge-base root resolution for the pre-Hopper kernel wiki scripts.

The knowledge base is a flat repository — SKILL.md, data/, wiki/, sources/,
queries/ all live at the same root. By default the root is this file's
grandparent directory:

    <kb-root>/scripts/_wiki_root.py  ->  <kb-root>

No environment variable is required for the common case. An optional
PREHOPPER_WIKI_ROOT override is honored for advanced setups (e.g. running the
scripts from a separate checkout). Any resolved root is validated by checking
for `data/tags.yaml` and `wiki/`; a misconfigured override hard-errors rather
than silently returning a wrong directory.

Design lineage: the resolution behavior (env override -> script-grandparent ->
walk-up autodetect -> hard error) follows the reference knowledge base's
pattern, reimplemented here with a pre-Hopper-neutral environment variable.
There is intentionally no dependency on any Blackwell-named variable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Environment variable that, when set, overrides root autodetection.
ENV_VAR = "PREHOPPER_WIKI_ROOT"


def _looks_like_wiki_root(p: Path) -> bool:
    """A directory is a knowledge-base root iff it has data/tags.yaml + wiki/."""
    return (p / "data" / "tags.yaml").is_file() and (p / "wiki").is_dir()


def _error(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def resolve_wiki_root() -> Path:
    # 1. Explicit env override (advanced use).
    env = os.environ.get(ENV_VAR)
    if env:
        p = Path(env).expanduser().resolve()
        if _looks_like_wiki_root(p):
            return p
        _error(
            f"{ENV_VAR}={env!r} does not point at a valid pre-Hopper kernel "
            f"wiki root (missing data/tags.yaml or wiki/)."
        )

    # 2. Default: this script's grandparent == repository/knowledge-base root.
    default_root = Path(__file__).resolve().parent.parent
    if _looks_like_wiki_root(default_root):
        return default_root

    # 3. Autodetect: walk up from the script location and from the cwd.
    seen: set[Path] = set()
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if _looks_like_wiki_root(candidate):
                return candidate

    _error(
        "Could not locate the pre-Hopper kernel wiki root.\n"
        "       Expected a directory containing `data/tags.yaml` and `wiki/`.\n"
        f"       Fix: run scripts from inside the knowledge-base directory, or\n"
        f"       set {ENV_VAR} to its absolute path."
    )
    return Path()  # unreachable


WIKI_ROOT = resolve_wiki_root()
