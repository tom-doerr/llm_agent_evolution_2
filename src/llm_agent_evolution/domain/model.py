import uuid
from dataclasses import dataclass
from typing import Optional, Literal, Dict

# Constants from spec
MAX_CHARS = 1000  # Maximum characters for chromosomes
MAX_POPULATION_SIZE = 1000000  # Default population size limit
TARGET_LENGTH = 23  # Target length for task optimization
MAX_OUTPUT_TOKENS = 40  # Limit token output for the DSPy LM
CHROMOSOME_TYPES = Literal["task", "mate_selection", "mutation"]

@dataclass
class Chromosome:
    content: str
    type: CHROMOSOME_TYPES
    
    def __post_init__(self):
        if len(self.content) > MAX_CHARS:
            self.content = self.content[:MAX_CHARS]

@dataclass
class Agent:
    task_chromosome: Chromosome
    mate_selection_chromosome: Chromosome
    mutation_chromosome: Chromosome
    id: str = None
    reward: Optional[float] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    @property
    def chromosomes(self) -> Dict[str, Chromosome]:
        return {
            "task": self.task_chromosome,
            "mate_selection": self.mate_selection_chromosome,
            "mutation": self.mutation_chromosome
        }
    
    def get_chromosome(self, type_: CHROMOSOME_TYPES) -> Chromosome:
        return self.chromosomes[type_]
