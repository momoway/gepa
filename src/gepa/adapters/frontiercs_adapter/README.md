# Frontier-CS Adapter

GEPA adapter for the [Frontier-CS](https://github.com/gepa-ai/gepa) algorithmic competitive programming benchmark. Optimizes the system prompt used to instruct an LLM to generate C++ solutions for algorithmic problems.

## Prerequisites

1. **Judge server**: The Frontier-CS judge must be running. From the algorithmic directory:

   ```bash
   cd /path/to/Frontier-CS-synthetic/Frontier-CS/algorithmic
   docker-compose up -d
   ```

2. **Dependencies**: `requests`, `litellm` (via `gepa[full]`)

## Dataset

Load problems from the local `problems/` directory:

```python
from gepa.examples.frontiercs import load_all_problems

problems = load_all_problems(
    problems_dir="/data/hry/Frontier-CS-synthetic/Frontier-CS/algorithmic/problems"
)
```

Each data instance has `problem_id` and `statement` (from `statement.txt`).

## Usage

```python
import gepa
from gepa.adapters.frontiercs_adapter import FrontierCSAdapter

problems = gepa.examples.frontiercs.load_all_problems(
    problems_dir="/data/hry/Frontier-CS-synthetic/Frontier-CS/algorithmic/problems"
)

seed_prompt = {
    "system_prompt": (
        "You are a competitive programmer. You will be given a problem statement. "
        "Implement a solution in C++. Wrap the code in ```cpp and ```. "
        "Your response should ONLY contain the C++ code."
    )
}

result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=problems[:5],
    valset=problems[5:8],
    adapter=FrontierCSAdapter(
        model="openai/gpt-4o",
        judge_url="http://localhost:8081",
    ),
    reflection_lm="openai/gpt-4o",
    max_metric_calls=50,
)
```

## Components

- **Candidate**: `system_prompt` — the instruction for generating C++ code
- **Evaluation**: LLM generates code → submit to judge → score (0–100, normalized to 0–1)
- **Reflection**: Judge feedback (score, case results, errors) is used to improve the prompt
