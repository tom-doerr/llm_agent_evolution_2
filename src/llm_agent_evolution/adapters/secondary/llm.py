import dspy
from typing import List, Optional, Any
from llm_agent_evolution.domain.model import Agent, Chromosome
from llm_agent_evolution.ports.secondary import LLMPort

# Constants from spec
MAX_OUTPUT_TOKENS = 40  # Limit token output for the DSPy LM

class DSPyLLMAdapter(LLMPort):
    """Adapter for LLM interactions using DSPy"""
    
    def __init__(self, model_name: str = "openrouter/google/gemini-2.0-flash-001"):
        """Initialize the LLM adapter with the specified model"""
        self.lm = dspy.LM(model_name)
        self.eval_command = None  # Will be set by the application
    
    def generate_mutation(self, chromosome: Chromosome, mutation_instructions: str) -> Chromosome:
        """Generate a mutation for a chromosome based on instructions"""
        try:
            # Create prompt for mutation
            prompt = self._create_mutation_prompt(chromosome, mutation_instructions)
            
            # Generate the mutation
            response = self.lm(prompt, max_tokens=MAX_OUTPUT_TOKENS)
            
            # Process the response
            response_text = self._process_llm_response(response)
            
            # Create and return the new chromosome
            return Chromosome(
                content=response_text.strip(),
                type=chromosome.type
            )
        except Exception as e:
            print(f"Error generating mutation: {e}")
            # Return a copy of the original chromosome if mutation fails
            return Chromosome(
                content=chromosome.content,
                type=chromosome.type
            )
    
    def _create_mutation_prompt(self, chromosome: Chromosome, mutation_instructions: str) -> str:
        """Create a prompt for mutation"""
        return f"""
        You are modifying a chromosome for an AI agent.
        
        Original chromosome content:
        {chromosome.content}
        
        Mutation instructions:
        {mutation_instructions}
        
        Provide only the modified chromosome content:
        """
    
    def _process_llm_response(self, response: Any) -> str:
        """Process the LLM response into a string"""
        # Handle different response types (DSPy might return a list)
        if isinstance(response, list):
            return " ".join(str(item) for item in response)
        else:
            return str(response)
    
    def select_mate(self, agent: Agent, candidates: List[Agent]) -> Agent:
        """Select a mate from candidates based on agent's mate selection chromosome"""
        if not candidates:
            return None
        
        try:
            # Create prompt for mate selection
            prompt = self._create_mate_selection_prompt(agent, candidates)
            
            # Generate the selection
            response = self.lm(prompt, max_tokens=5)
            
            # Process the response
            response_text = self._process_llm_response(response)
            
            # Parse the selection
            return self._parse_mate_selection(response_text, candidates)
        except Exception as e:
            print(f"Error selecting mate: {e}")
            # Return a random candidate if selection fails
            import random
            return random.choice(candidates)
    
    def _create_mate_selection_prompt(self, agent: Agent, candidates: List[Agent]) -> str:
        """Create a prompt for mate selection"""
        # Create a prompt with candidate DNA information as specified in the spec
        candidates_info = "\n\n".join([
            f"Candidate {i+1} DNA:\n"
            f"ID: {candidate.id}\n"
            f"Task Chromosome: {candidate.task_chromosome.content}\n"
            f"Mate Selection Chromosome: {candidate.mate_selection_chromosome.content}\n"
            f"Mutation Chromosome: {candidate.mutation_chromosome.content}\n"
            f"Reward: {candidate.reward}"
            for i, candidate in enumerate(candidates)
        ])
        
        return f"""
        You are selecting a mate for an AI agent based on DNA analysis.
        
        Your DNA contains these mate selection instructions:
        {agent.mate_selection_chromosome.content}
        
        Available candidate DNA:
        {candidates_info}
        
        Select one candidate by returning ONLY the number (1-{len(candidates)}) of your choice:
        """
    
    def _parse_mate_selection(self, response_text: str, candidates: List[Agent]) -> Agent:
        """Parse the mate selection response"""
        try:
            # First try to parse as a direct number
            selection = int(response_text.strip()) - 1
            if 0 <= selection < len(candidates):
                return candidates[selection]
                
            # If out of range, try to find any number in the response
            import re
            numbers = re.findall(r'\b(\d+)\b', response_text)
            for num in numbers:
                idx = int(num) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
                    
            # If still no valid selection, use a weighted approach based on rewards
            if candidates:
                # Weight by reward (with a minimum weight to avoid zero probabilities)
                weights = [max(0.1, c.reward) if c.reward is not None else 0.1 for c in candidates]
                total = sum(weights)
                if total > 0:
                    # Normalize weights
                    weights = [w/total for w in weights]
                    # Select based on weights
                    import random
                    return random.choices(candidates, weights=weights, k=1)[0]
                
            # Fallback to random selection
            import random
            return random.choice(candidates)
        except (ValueError, TypeError, IndexError):
            # If parsing fails, use a weighted approach
            if candidates:
                # Weight by reward (with a minimum weight to avoid zero probabilities)
                weights = [max(0.1, c.reward) if c.reward is not None else 0.1 for c in candidates]
                total = sum(weights)
                if total > 0:
                    # Normalize weights
                    weights = [w/total for w in weights]
                    # Select based on weights
                    import random
                    return random.choices(candidates, weights=weights, k=1)[0]
            
            # Fallback to random selection
            import random
            return random.choice(candidates)
    
    def evaluate_task_output(self, output: str) -> float:
        """
        Evaluate the output using the specified evaluation command
        If no command is set, returns 0
        """
        if not self.eval_command:
            print("Warning: No evaluation command set")
            return 0
            
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
            return 0
