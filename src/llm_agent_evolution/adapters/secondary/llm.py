import dspy
import random
import re
import tempfile
import os
import subprocess
from typing import List, Any
from llm_agent_evolution.domain.model import Agent, Chromosome, MAX_OUTPUT_TOKENS

class DSPyLLMAdapter:
    def __init__(self, model_name: str = "openrouter/google/gemini-2.0-flash-001"):
        self.lm = dspy.LM(model_name)
        self.eval_command = None
    
    def generate_mutation(self, chromosome: Chromosome, mutation_instructions: str) -> Chromosome:
        # We don't actually use this for mutation as per spec
        # Just return the original chromosome
        return chromosome
    
    def _process_llm_response(self, response: Any) -> str:
        if isinstance(response, list):
            return " ".join(str(item) for item in response)
        else:
            return str(response)
    
    def select_mate(self, agent: Agent, candidates: List[Agent]) -> Agent:
        if not candidates:
            return None
        
        try:
            candidates_info = "\n\n".join([
                f"Candidate {i+1}:\n"
                f"ID: {candidate.id}\n"
                f"Task: {candidate.task_chromosome.content[:50]}...\n"
                f"Reward: {candidate.reward}"
                for i, candidate in enumerate(candidates)
            ])
            
            prompt = f"""
            You are selecting a mate for an AI agent.
            
            Your mate selection instructions:
            {agent.mate_selection_chromosome.content}
            
            Available candidates:
            {candidates_info}
            
            Select one candidate by returning ONLY the number (1-{len(candidates)}) of your choice:
            """
            
            response = self.lm(prompt, max_tokens=5)
            response_text = self._process_llm_response(response)
            
            try:
                selection = int(response_text.strip()) - 1
                if 0 <= selection < len(candidates):
                    return candidates[selection]
                    
                numbers = re.findall(r'\b(\d+)\b', response_text)
                for num in numbers:
                    idx = int(num) - 1
                    if 0 <= idx < len(candidates):
                        return candidates[idx]
            except (ValueError, TypeError, IndexError):
                pass
                
            # Fallback to weighted selection
            weights = [max(0.1, c.reward) if c.reward is not None else 0.1 for c in candidates]
            total = sum(weights)
            if total > 0:
                weights = [w/total for w in weights]
                return random.choices(candidates, weights=weights, k=1)[0]
            
            return random.choice(candidates)
        except Exception as e:
            print(f"Error selecting mate: {e}")
            return random.choice(candidates)
    
    def evaluate_task_output(self, output: str) -> float:
        if not self.eval_command:
            print("Warning: No evaluation command set")
            return 0
            
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
                capture_output=True,
                env=os.environ
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
            return 0
