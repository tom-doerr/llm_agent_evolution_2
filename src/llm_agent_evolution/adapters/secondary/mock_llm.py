import random
from typing import List, Optional
from ....domain.model import Agent, Chromosome
from ....ports.secondary import LLMPort

class MockLLMAdapter(LLMPort):
    """Mock adapter for LLM interactions for testing purposes"""
    
    def __init__(self, seed: int = None):
        """Initialize the mock LLM adapter with an optional random seed"""
        if seed is not None:
            random.seed(seed)
    
    def generate_mutation(self, chromosome: Chromosome, mutation_instructions: str) -> Chromosome:
        """Generate a mock mutation for a chromosome"""
        # For task chromosomes, generate strings with 'a's to test the hidden goal
        if chromosome.type == "task":
            # Generate random number of 'a's (0-23)
            a_count = random.randint(0, 23)
            content = 'a' * a_count
            
            # Sometimes add extra characters
            if random.random() < 0.3:
                extra_chars = random.randint(1, 10)
                content += 'x' * extra_chars
        else:
            # For other chromosome types, generate simple instructions
            options = [
                "Select the candidate with the highest reward",
                "Choose a mate with the most 'a's in their task chromosome",
                "Prefer candidates with shorter chromosomes",
                "Try to add more 'a' characters to the chromosome",
                "Keep the chromosome short and focused"
            ]
            content = random.choice(options)
        
        return Chromosome(
            content=content,
            type=chromosome.type
        )
    
    def select_mate(self, agent: Agent, candidates: List[Agent]) -> Agent:
        """Select a mate from candidates randomly"""
        if not candidates:
            return None
        
        return random.choice(candidates)
    
    def evaluate_task_output(self, output: str) -> float:
        """
        Evaluate the output based on the hidden goal:
        - Reward increases for every 'a' for the first 23 characters
        - Reward decreases for every character after 23 characters
        """
        # Count 'a's in the first 23 characters
        a_count = output[:23].count('a')
        
        # Penalty for exceeding 23 characters
        length_penalty = max(0, len(output) - 23)
        
        # Calculate reward
        reward = a_count - length_penalty
        
        return reward
