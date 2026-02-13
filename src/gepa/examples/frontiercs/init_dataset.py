# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from pathlib import Path

from gepa.adapters.frontiercs_adapter import FrontierCSDataInst


def load_all_problems(
    problems_dir: str | Path = "/data/hry/Frontier-CS-synthetic/Frontier-CS/algorithmic/problems",
) -> list[FrontierCSDataInst]:
    """Load all Frontier-CS problems as a flat list (no train/val/test split)."""
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
