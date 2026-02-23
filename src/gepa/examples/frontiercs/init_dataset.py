# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import os
import subprocess
from pathlib import Path

from gepa.adapters.frontiercs_adapter import FrontierCSDataInst

FRONTIERCS_REPO = "https://github.com/FrontierCS/Frontier-CS.git"


def get_frontiercs_problems_dir() -> Path:
    """Return path to Frontier-CS algorithmic problems, cloning the repo if needed.

    Uses cache dir: $XDG_CACHE_HOME/gepa/frontier-cs or ~/.cache/gepa/frontier-cs
    """
    cache_base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    repo_dir = Path(cache_base) / "gepa" / "frontier-cs"
    problems_dir = repo_dir / "algorithmic" / "problems"

    if not problems_dir.exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        if not (repo_dir / ".git").exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", FRONTIERCS_REPO, str(repo_dir)],
                check=True,
                capture_output=True,
            )
        else:
            subprocess.run(
                ["git", "pull", "-q"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
            )

    return problems_dir


def load_all_problems(
    problems_dir: str | Path | None = None,
) -> list[FrontierCSDataInst]:
    """Load all Frontier-CS problems as a flat list (no train/val/test split).

    If problems_dir is None or empty, clones/fetches from https://github.com/FrontierCS/Frontier-CS.git
    to ~/.cache/gepa/frontier-cs and uses algorithmic/problems.
    """
    if not problems_dir:
        problems_path = get_frontiercs_problems_dir()
    else:
        problems_path = Path(problems_dir)
    if not problems_path.exists():
        raise FileNotFoundError(f"Problems directory not found: {problems_path}")

    items: list[FrontierCSDataInst] = []
    for pdir in sorted(problems_path.iterdir(), key=lambda p: (len(p.name), p.name)):
        if not pdir.is_dir():
            continue
        statement_file = pdir / "statement.txt"
        if not statement_file.exists():
            continue
        problem_id = pdir.name
        statement = statement_file.read_text(encoding="utf-8", errors="replace")
        items.append({"problem_id": problem_id, "statement": statement})

    if not items:
        raise ValueError(f"No valid problems found in {problems_path}")

    return items
