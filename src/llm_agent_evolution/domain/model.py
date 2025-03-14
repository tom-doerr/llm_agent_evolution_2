import uuid
from dataclasses import dataclass
from typing import Optional, Literal, List, Dict, Any

# Constants from spec
MAX_CHARS = 1000  # Maximum characters for chromosomes
MAX_POPULATION_SIZE = 1000000  # Default population size limit
CHROMOSOME_TYPES = Literal["task", "mate_selection", "mutation"]

@dataclass
class Chromosome:
    # A chromosome with content and type
    content: str
    type: CHROMOSOME_TYPES
    
    def __post_init__(self):
        # Ensure chromosome doesn't exceed max length
        if len(self.content) > MAX_CHARS:
            self.content = self.content[:MAX_CHARS]

@dataclass
class Agent:
    # An agent with three chromosomes and a reward score
    task_chromosome: Chromosome
    mate_selection_chromosome: Chromosome
    mutation_chromosome: Chromosome
    id: str = None
    reward: Optional[float] = None
    
    def __post_init__(self):
        # Generate ID if not provided
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    @property
    def chromosomes(self) -> Dict[str, Chromosome]:
        """Return all chromosomes as a dictionary"""
        return {
            "task": self.task_chromosome,
            "mate_selection": self.mate_selection_chromosome,
            "mutation": self.mutation_chromosome
        }
    
    def get_chromosome(self, type_: CHROMOSOME_TYPES) -> Chromosome:
        """Get a specific chromosome by type"""
        return self.chromosomes[type_]
