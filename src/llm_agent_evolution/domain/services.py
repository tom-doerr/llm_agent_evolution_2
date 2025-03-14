import random
from typing import List, Tuple, Optional
import numpy as np
from .model import Agent, Chromosome

# Constants
CHROMOSOME_SWITCH_PROBABILITY = 0.3  # Probability of switching chromosomes at hotspots
HOTSPOT_CHARS = ".,;:!?()[]{}'\"\n "  # Punctuation and spaces as hotspots
TARGET_LENGTH = 23  # Target length for task optimization (used in combine_chromosomes)

def select_parents_pareto(population: List[Agent], num_parents: int) -> List[Agent]:
    """
    Select parents using Pareto distribution weighting by fitness^2
    Uses weighted sampling without replacement
    
    Improved to better handle negative rewards and ensure diversity
    """
    if not population or num_parents <= 0:
        return []
    
    # Filter out agents without rewards
    valid_agents = [agent for agent in population if agent.reward is not None]
    if not valid_agents:
        return random.sample(population, min(num_parents, len(population)))
    
    # Get reward statistics
    rewards = [agent.reward for agent in valid_agents]
    min_reward = min(rewards)
    max_reward = max(rewards)
    reward_range = max(1.0, max_reward - min_reward)  # Avoid division by zero
    
    # Normalize rewards to [0,1] range and then square them for Pareto distribution
    # Add a small epsilon to ensure even worst agents have some chance
    epsilon = 0.01
    adjusted_rewards = [((agent.reward - min_reward) / reward_range + epsilon) ** 2 
                        for agent in valid_agents]
    
    # Normalize weights
    total_weight = sum(adjusted_rewards)
    weights = [w / total_weight for w in adjusted_rewards]
    
    # Weighted sampling without replacement
    selected_indices = np.random.choice(
        len(valid_agents), 
        size=min(num_parents, len(valid_agents)), 
        replace=False, 
        p=weights
    )
    
    selected_parents = [valid_agents[i] for i in selected_indices]
    
    # Ensure diversity by adding some random agents if we have enough parents
    if len(selected_parents) < num_parents and len(valid_agents) > len(selected_parents):
        remaining_agents = [a for a in valid_agents if a not in selected_parents]
        random_parents = random.sample(
            remaining_agents, 
            min(num_parents - len(selected_parents), len(remaining_agents))
        )
        selected_parents.extend(random_parents)
    
    return selected_parents

def combine_chromosomes(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """
    Combine two chromosomes by switching at hotspots
    Designed to have approximately one chromosome jump per chromosome
    
    Enhanced to handle different length chromosomes and ensure more effective combinations
    """
    if not parent1.content and not parent2.content:
        return Chromosome(content="", type=parent1.type)
    
    if not parent1.content:
        return Chromosome(content=parent2.content, type=parent1.type)
    
    if not parent2.content:
        return Chromosome(content=parent1.content, type=parent1.type)
    
    # Determine which parent has more valuable content based on length
    # For task chromosomes, we prefer shorter content (up to a reasonable length)
    if parent1.type == "task":
        # For task chromosomes, prefer content closer to a reasonable length
        optimal_length = 100  # A reasonable target length for most tasks
        p1_value = max(0, optimal_length - abs(len(parent1.content) - optimal_length))
        p2_value = max(0, optimal_length - abs(len(parent2.content) - optimal_length))
        
        # If one parent is significantly better, bias toward it
        if p1_value > p2_value * 1.5:
            primary_parent, secondary_parent = parent1, parent2
        elif p2_value > p1_value * 1.5:
            primary_parent, secondary_parent = parent2, parent1
        else:
            # Otherwise choose randomly
            if random.random() < 0.5:
                primary_parent, secondary_parent = parent1, parent2
            else:
                primary_parent, secondary_parent = parent2, parent1
    else:
        # For other chromosome types, randomly select primary parent
        if random.random() < 0.5:
            primary_parent, secondary_parent = parent1, parent2
        else:
            primary_parent, secondary_parent = parent2, parent1
    
    # Find all hotspots in primary parent's content
    hotspots = [i for i, char in enumerate(primary_parent.content) if char in HOTSPOT_CHARS]
    if not hotspots:
        # If no hotspots, add some arbitrary points
        content_len = len(primary_parent.content)
        hotspots = [i for i in range(0, content_len, max(1, content_len // 5))]
    
    # Start with primary parent's content
    result = list(primary_parent.content)
    using_primary = True
    
    # Calculate number of switches based on content length
    content_len = len(primary_parent.content)
    target_switches = max(1, content_len // 100)  # Aim for ~1 switch per 100 chars
    
    # Sort hotspots and select a subset for switching
    hotspots.sort()
    if len(hotspots) > 0:
        switch_points = sorted(random.sample(hotspots, min(target_switches, len(hotspots))))
        
        # Perform the switches
        for point in switch_points:
            if random.random() < CHROMOSOME_SWITCH_PROBABILITY:
                using_primary = not using_primary
                
                if not using_primary and point < len(secondary_parent.content):
                    # Calculate how much content to take from secondary parent
                    if point < len(secondary_parent.content):
                        # Take content from secondary parent from this point
                        secondary_content = list(secondary_parent.content[point:])
                        
                        # Adjust result length
                        if point + len(secondary_content) > len(result):
                            result = result[:point] + secondary_content
                        else:
                            result[point:point+len(secondary_content)] = secondary_content
    
    # For task chromosomes, ensure we don't exceed a reasonable length
    if parent1.type == "task":
        combined_content = ''.join(result)
        max_length = 1000  # Use a reasonable maximum length
        if len(combined_content) > max_length:
            # Truncate to a reasonable length
            combined_content = combined_content[:max_length]
        return Chromosome(content=combined_content, type=parent1.type)
    
    return Chromosome(content=''.join(result), type=parent1.type)

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
