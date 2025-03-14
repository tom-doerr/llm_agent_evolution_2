# LLM Agent Evolution Implementation Plan

## Architecture: Hexagonal (Ports and Adapters)

We'll implement the LLM Agent Evolution project using Hexagonal Architecture to separate core domain logic from external concerns, making the system more maintainable, testable, and flexible.

## 1. Project Structure

```
src/llm_agent_evolution/
├── __init__.py
├── domain/                 # Core domain logic
│   ├── __init__.py
│   ├── model.py            # Domain entities (Agent, Chromosome)
│   └── services.py         # Domain services (evolution logic)
├── ports/                  # Interface definitions
│   ├── __init__.py
│   ├── primary.py          # Driving ports (use cases)
│   └── secondary.py        # Driven ports (LLM, logging, statistics)
├── adapters/               # Implementation of ports
│   ├── __init__.py
│   ├── primary/            # Driving adapters
│   │   ├── __init__.py
│   │   └── cli.py          # Command-line interface
│   └── secondary/          # Driven adapters
│       ├── __init__.py
│       ├── llm.py          # LLM interaction
│       ├── logging.py      # Logging to file
│       └── statistics.py   # Statistics tracking
└── application.py          # Application configuration and wiring
```

## 2. Implementation Phases

### Phase 1: Core Domain
- Define domain entities (Agent, Chromosome)
- Implement core evolution services
- Create pure functions for chromosome combination, parent selection

### Phase 2: Ports
- Define primary ports (evolution use cases)
- Define secondary ports (LLM, logging, statistics interfaces)

### Phase 3: Adapters
- Implement LLM adapter using DSPy
- Implement file logging adapter
- Implement statistics tracking adapter
- Implement CLI adapter

### Phase 4: Application Configuration
- Wire everything together
- Configure dependency injection
- Set up multithreading

### Phase 5: Testing
- Unit tests for domain logic
- Integration tests with mock adapters
- End-to-end tests

## 3. Key Components

### Domain Model

```python
class Chromosome:
    def __init__(self, content, type_):
        self.content = content
        self.type = type_  # "task", "mate_selection", or "mutation"

class Agent:
    def __init__(self, task_chromosome, mate_selection_chromosome, mutation_chromosome, id=None):
        self.id = id or str(uuid.uuid4())
        self.task_chromosome = task_chromosome
        self.mate_selection_chromosome = mate_selection_chromosome
        self.mutation_chromosome = mutation_chromosome
        self.reward = None
```

### Primary Ports (Use Cases)

```python
class EvolutionUseCase:
    """Interface for driving the evolution process"""
    def initialize_population(self, size): pass
    def select_parents(self, population, num_parents): pass
    def mate_agents(self, parent1, parent2): pass
    def evaluate_agent(self, agent): pass
    def add_to_population(self, population, agent): pass
    def get_population_stats(self): pass
```

### Secondary Ports

```python
class LLMPort:
    """Interface for LLM interactions"""
    def generate_mutation(self, chromosome, mutation_instructions): pass
    def select_mate(self, candidates, selection_criteria): pass

class LoggingPort:
    """Interface for logging"""
    def initialize_log(self): pass
    def log_evaluation(self, agent_id, reward, chromosomes): pass
    def log_population_stats(self, stats): pass

class StatisticsPort:
    """Interface for statistics tracking"""
    def track_reward(self, reward): pass
    def get_stats(self): pass
```

## 4. Multithreading Strategy

- Use a thread pool with configurable number of worker threads
- Each worker continuously:
  1. Selects parents
  2. Creates new agent through mating
  3. Evaluates the new agent
  4. Adds to population
- Use thread-safe data structures for population management
- Implement sliding window statistics with thread-safe updates

## 5. Evaluation Task

For the specific task mentioned in the spec:
- Reward increases for every 'a' for the first 23 characters
- Reward decreases for every character after 23 characters
- Limit token output to 40

We'll implement this as a specific adapter without revealing the goal to the agents.

## 6. Dependencies

- DSPy for LLM interaction
- Standard library for most functionality
- Rich for console output (optional)
- Threading for parallelism

## 7. Next Steps

1. Set up basic project structure
2. Implement domain model
3. Define ports
4. Create minimal working adapters
5. Wire everything together for a simple end-to-end test
6. Implement multithreading
7. Add statistics and logging
8. Refine and optimize

## 8. Success Criteria

- System evolves agents that discover the hidden goal
- Multithreading works correctly
- Statistics are tracked and displayed
- Detailed logging to file
- Clean separation of concerns via hexagonal architecture
