import random
from typing import List, Optional
from llm_agent_evolution.domain.model import Agent, Chromosome
from llm_agent_evolution.ports.secondary import LLMPort

class MockLLMAdapter(LLMPort):
    """Mock adapter for LLM interactions for testing purposes"""
    
    def __init__(self, seed: int = None):
        """Initialize the mock LLM adapter with an optional random seed"""
        if seed is not None:
            random.seed(seed)
        self.eval_command = None  # Will be set by the application
    
    def generate_mutation(self, chromosome: Chromosome, mutation_instructions: str) -> Chromosome:
        """Generate a mock mutation for a chromosome"""
        # For task chromosomes, generate random content without leaking the task
        if chromosome.type == "task":
            # Generate random content
            options = [
                "Hello world",
                "Testing the mutation",
                "This is a sample output",
                "Random text for evaluation",
                "The quick brown fox jumps over the lazy dog"
            ]
            content = random.choice(options)
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
        Evaluate the output using the specified evaluation command
        If no command is set, returns a random score for testing
        """
        if not self.eval_command:
            # For testing, return a random score
            return random.uniform(0, 10)
            
        # Use the script evaluator to run the command
        from llm_agent_evolution.adapters.secondary.script_evaluator import ScriptEvaluatorAdapter
        evaluator = ScriptEvaluatorAdapter()
        
        try:
            # Create a temporary script that runs the eval command
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
                script_path = f.name
                f.write("#!/bin/sh\n")
                f.write(f"{self.eval_command}\n")
            
            # Make executable
            os.chmod(script_path, 0o755)
            
            # Evaluate
            reward = evaluator.evaluate(output, script_path)
            
            # Clean up
            os.remove(script_path)
            
            return reward
        except Exception as e:
            print(f"Evaluation error: {e}")
            return random.uniform(0, 5)  # Fallback for testing
