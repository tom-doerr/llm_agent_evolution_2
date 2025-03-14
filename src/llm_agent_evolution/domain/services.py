import random
from typing import List, Tuple, Optional
import numpy as np
from .model import Agent, Chromosome

# Constants
CHROMOSOME_SWITCH_PROBABILITY = 0.3  # Probability of switching chromosomes at hotspots
HOTSPOT_CHARS = ".,;:!?()[]{}'\"\n "  # Punctuation and spaces as hotspots

def select_parents_pareto(population: List[Agent], num_parents: int) -> List[Agent]:
    """
    Select parents using Pareto distribution weighting by fitness^2
    Uses weighted sampling without replacement
    """
    if not population or num_parents <= 0:
        return []
    
    # Filter out agents without rewards
    valid_agents = [agent for agent in population if agent.reward is not None]
    if not valid_agents:
        return random.sample(population, min(num_parents, len(population)))
    
    # Square the rewards for Pareto distribution and handle negative rewards
    min_reward = min(agent.reward for agent in valid_agents)
    adjusted_rewards = [(agent.reward - min_reward + 1) ** 2 for agent in valid_agents]
    
    # Normalize weights
    total_weight = sum(adjusted_rewards)
    if total_weight == 0:
        # If all weights are zero, use uniform distribution
        return random.sample(valid_agents, min(num_parents, len(valid_agents)))
    
    weights = [w / total_weight for w in adjusted_rewards]
    
    # Weighted sampling without replacement
    selected_indices = np.random.choice(
        len(valid_agents), 
        size=min(num_parents, len(valid_agents)), 
        replace=False, 
        p=weights
    )
    
    return [valid_agents[i] for i in selected_indices]

def combine_chromosomes(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """
    Combine two chromosomes by switching at hotspots
    Designed to have approximately one chromosome jump per chromosome
    """
    if not parent1.content or not parent2.content:
        return Chromosome(
            content=parent1.content or parent2.content,
            type=parent1.type
        )
    
    # Find all hotspots in parent1's content
    hotspots = [i for i, char in enumerate(parent1.content) if char in HOTSPOT_CHARS]
    if not hotspots:
        # If no hotspots, add some arbitrary points
        content_len = len(parent1.content)
        hotspots = [i for i in range(0, content_len, max(1, content_len // 5))]
    
    # Start with parent1's content
    result = list(parent1.content)
    current_parent = 1
    
    # Calculate number of switches based on content length to average one switch per chromosome
    content_len = len(parent1.content)
    target_switches = max(1, content_len // 100)  # Aim for ~1 switch per 100 chars
    
    # Sort hotspots and select a subset for switching
    hotspots.sort()
    switch_points = sorted(random.sample(hotspots, min(target_switches, len(hotspots))))
    
    # Perform the switches
    for point in switch_points:
        if random.random() < CHROMOSOME_SWITCH_PROBABILITY:
            current_parent = 3 - current_parent  # Toggle between 1 and 2
            if current_parent == 2:
                # Switch to parent2's content from this point
                remaining = len(parent2.content) - point
                if remaining > 0:
                    result[point:] = list(parent2.content[point:min(len(parent2.content), point + len(result) - point)])
    
    return Chromosome(
        content=''.join(result),
        type=parent1.type
    )

def mate_agents(parent1: Agent, parent2: Agent) -> Agent:
    """
    Create a new agent by combining chromosomes from two parents
    Each chromosome has a chance to be combined or taken whole
    """
    # Combine each chromosome type
    task_chromosome = combine_chromosomes(
        parent1.task_chromosome, 
        parent2.task_chromosome
    )
    
    mate_selection_chromosome = combine_chromosomes(
        parent1.mate_selection_chromosome,
        parent2.mate_selection_chromosome
    )
    
    mutation_chromosome = combine_chromosomes(
        parent1.mutation_chromosome,
        parent2.mutation_chromosome
    )
    
    # Create and return new agent
    return Agent(
        task_chromosome=task_chromosome,
        mate_selection_chromosome=mate_selection_chromosome,
        mutation_chromosome=mutation_chromosome
    )
