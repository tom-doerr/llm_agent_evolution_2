# LLM Agent Evolution

Framework for evolving LLM-based agents through evolutionary algorithms.

## Installation

```bash
pip install -e .
```

## Basic Usage

```python
from llm_agent_evolution import evolve_agents

# Example evolution setup
population = evolve_agents(
    population_size=100,
    mutation_rate=0.1,
    selection_pressure=0.2
)
```

## Development Setup

```bash
# Install with test dependencies
pip install -e ".[test]"

# Run tests
python -m pytest
```
