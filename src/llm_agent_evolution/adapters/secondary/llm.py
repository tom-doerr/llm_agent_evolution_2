import dspy
from typing import List, Optional
from llm_agent_evolution.domain.model import Agent, Chromosome
from llm_agent_evolution.ports.secondary import LLMPort

# Constants from spec
MAX_OUTPUT_TOKENS = 40  # Limit token output for the DSPy LM
TARGET_LENGTH = 23  # Target length for the hidden goal

class DSPyLLMAdapter(LLMPort):
    """Adapter for LLM interactions using DSPy"""
    
    def __init__(self, model_name: str = "openrouter/google/gemini-2.0-flash-001"):
        """Initialize the LLM adapter with the specified model"""
        self.lm = dspy.LM(model_name)
    
    def generate_mutation(self, chromosome: Chromosome, mutation_instructions: str) -> Chromosome:
        """Generate a mutation for a chromosome based on instructions"""
        prompt = f"""
        You are modifying a chromosome for an AI agent.
        
        Original chromosome content:
        {chromosome.content}
        
        Mutation instructions:
        {mutation_instructions}
        
        Provide only the modified chromosome content:
        """
        
        # Generate the mutation
        response = self.lm(prompt, max_tokens=MAX_OUTPUT_TOKENS)
        
        # Handle different response types (DSPy might return a list)
        if isinstance(response, list):
            response_text = " ".join(str(item) for item in response)
        else:
            response_text = str(response)
        
        # Create and return the new chromosome
        return Chromosome(
            content=response_text.strip(),
            type=chromosome.type
        )
    
    def select_mate(self, agent: Agent, candidates: List[Agent]) -> Agent:
        """Select a mate from candidates based on agent's mate selection chromosome"""
        if not candidates:
            return None
        
        # Create a prompt with candidate information
        candidates_info = "\n\n".join([
            f"Candidate {i+1} ID: {candidate.id}\n"
            f"Task Chromosome: {candidate.task_chromosome.content}\n"
            f"Mate Selection Chromosome: {candidate.mate_selection_chromosome.content}\n"
            f"Mutation Chromosome: {candidate.mutation_chromosome.content}\n"
            f"Reward: {candidate.reward}"
            for i, candidate in enumerate(candidates)
        ])
        
        prompt = f"""
        You are selecting a mate for an AI agent.
        
        Your mate selection criteria:
        {agent.mate_selection_chromosome.content}
        
        Available candidates:
        {candidates_info}
        
        Select one candidate by returning ONLY the number (1-{len(candidates)}) of your choice:
        """
        
        # Generate the selection
        response = self.lm(prompt, max_tokens=5)
        
        # Handle different response types (DSPy might return a list)
        if isinstance(response, list):
            response_text = " ".join(str(item) for item in response)
        else:
            response_text = str(response)
        
        try:
            # Parse the response to get the selected candidate index
            selection = int(response_text.strip()) - 1
            if 0 <= selection < len(candidates):
                return candidates[selection]
            else:
                # If invalid selection, return a random candidate
                import random
                return random.choice(candidates)
        except (ValueError, TypeError):
            # If parsing fails, return a random candidate
            import random
            return random.choice(candidates)
    
    def evaluate_task_output(self, output: str) -> float:
        """
        Evaluate the output based on the hidden goal:
        - Reward increases for every 'a' for the first TARGET_LENGTH characters
        - Reward decreases for every character after TARGET_LENGTH characters
        """
        # Count 'a's in the first TARGET_LENGTH characters
        a_count = output[:TARGET_LENGTH].count('a')
        
        # Penalty for exceeding TARGET_LENGTH
        length_penalty = max(0, len(output) - TARGET_LENGTH)
        
        # Calculate reward
        reward = a_count - length_penalty
        
        # Print debug info for important cases
        if a_count > 20 or reward > 20:
            print(f"Debug - High reward output: '{output[:50]}{'...' if len(output) > 50 else ''}'")
            print(f"Debug - a_count: {a_count}, length_penalty: {length_penalty}, reward: {reward}")
        
        return reward
