import random
from typing import List
import subprocess
import tempfile
import os
from llm_agent_evolution.domain.model import Agent, Chromosome, TARGET_LENGTH

class MockLLMAdapter:
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        self.eval_command = None
    
    def generate_mutation(self, chromosome: Chromosome, mutation_instructions: str) -> Chromosome:
        if chromosome.type == "task":
            original = chromosome.content
            
            if not original:
                options = [
                    "Hello world",
                    "Testing the mutation",
                    "This is a sample output",
                    "Random text for evaluation",
                    "The quick brown fox jumps over the lazy dog"
                ]
                content = random.choice(options)
            else:
                modifications = [
                    lambda s: s + " " + random.choice(["extra", "more", "additional"]),
                    lambda s: s.replace(" ", " " + random.choice(["a", "b", "c"]) + " "),
                    lambda s: s[:len(s)//2] + random.choice(["X", "Y", "Z"]) + s[len(s)//2:],
                    lambda s: "".join(random.sample(s, len(s))) if len(s) > 1 else s,
                    lambda s: s * 2 if len(s) < 10 else s
                ]
                content = random.choice(modifications)(original)
        else:
            if random.random() < 0.7 and chromosome.content:
                content = chromosome.content
                if "reward" not in content:
                    content += " Consider the reward."
                if "diversity" not in content:
                    content += " Maintain diversity."
            else:
                options = [
                    "Select the candidate with the highest reward",
                    "Choose a mate with diverse characteristics",
                    "Prefer candidates with better performance",
                    "Try to improve the chromosome with each mutation",
                    "Keep the chromosome focused on the evaluation criteria"
                ]
                content = random.choice(options)
        
        return Chromosome(
            content=content,
            type=chromosome.type
        )
    
    def select_mate(self, agent: Agent, candidates: List[Agent]) -> Agent:
        if not candidates:
            return None
        
        return random.choice(candidates)
    
    def evaluate_task_output(self, output: str) -> float:
        if not self.eval_command:
            return self._evaluate_hidden_goal(output)
        else:
            return self._evaluate_with_command(output)
    
    def _evaluate_hidden_goal(self, output: str) -> float:
        # Count only lowercase 'a' characters, not apostrophes or uppercase 'A'
        a_count = sum(1 for c in output[:TARGET_LENGTH] if c == 'a')
        length_penalty = max(0, len(output) - TARGET_LENGTH)
        reward = a_count - length_penalty
        return reward
    
    def _evaluate_with_command(self, output: str) -> float:
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
                script_path = f.name
                f.write("#!/bin/sh\n")
                f.write(f"{self.eval_command}\n")
            
            os.chmod(script_path, 0o755)
            
            process = subprocess.run(
                [script_path],
                input=output,
                text=True,
                capture_output=True
            )
            
            os.remove(script_path)
            
            if process.returncode != 0:
                print(f"Warning: Command failed: {process.stderr}")
                return 0.0
                
            try:
                return float(process.stdout.strip().split('\n')[-1])
            except (ValueError, IndexError):
                print(f"Warning: Could not parse output as float: {process.stdout}")
                return 0.0
                
        except Exception as e:
            print(f"Evaluation error: {e}")
            return len(output)
