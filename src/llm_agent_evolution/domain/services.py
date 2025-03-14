import random
from typing import List, Tuple, Optional
import numpy as np
from .model import Agent, Chromosome, MAX_CHARS

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
    
    # Handle case where all rewards are the same
    if min_reward == max_reward:
        return random.sample(valid_agents, min(num_parents, len(valid_agents)))
    
    # Shift rewards to positive range if needed
    if min_reward < 0:
        shift = abs(min_reward) + 1.0  # Add 1 to ensure all values are positive
        adjusted_rewards = [(r + shift) for r in rewards]
    else:
        # If all rewards are already positive, just use them directly
        adjusted_rewards = rewards.copy()
    
    # Square the rewards for Pareto distribution (emphasize higher rewards)
    squared_rewards = [r**2 for r in adjusted_rewards]
    
    # Normalize weights
    total_weight = sum(squared_rewards)
    weights = [w / total_weight for w in squared_rewards]
    
    # Weighted sampling without replacement
    selected_parents = []
    remaining_agents = valid_agents.copy()
    remaining_weights = weights.copy()
    
    # Select parents one by one
    for _ in range(min(num_parents, len(valid_agents))):
        if not remaining_agents:
            break
            
        # Normalize remaining weights
        weight_sum = sum(remaining_weights)
        if weight_sum <= 0:
            # If all weights are zero, select randomly
            idx = random.randrange(len(remaining_agents))
        else:
            normalized_weights = [w / weight_sum for w in remaining_weights]
            # Select based on weights
            idx = random.choices(range(len(remaining_agents)), weights=normalized_weights, k=1)[0]
        
        # Add selected agent to parents
        selected_parents.append(remaining_agents[idx])
        
        # Remove selected agent from candidates
        remaining_agents.pop(idx)
        remaining_weights.pop(idx)
    
    # Ensure diversity by adding some random agents if needed
    if len(selected_parents) < num_parents and len(population) > len(selected_parents):
        remaining_agents = [a for a in population if a not in selected_parents]
        # Add a small percentage of random agents for exploration
        num_random = max(1, int(num_parents * 0.2))
        random_parents = random.sample(
            remaining_agents, 
            min(num_random, len(remaining_agents))
        )
        selected_parents.extend(random_parents[:num_parents - len(selected_parents)])
    
    return selected_parents

def combine_chromosomes(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """
    Combine two chromosomes by switching at hotspots
    Designed to have approximately one chromosome jump per chromosome
    
    Enhanced to handle different length chromosomes and ensure more effective combinations
    """
    # Handle empty content cases
    if not parent1.content and not parent2.content:
        return Chromosome(content="", type=parent1.type)
    
    if not parent1.content:
        return Chromosome(content=parent2.content, type=parent1.type)
    
    if not parent2.content:
        return Chromosome(content=parent1.content, type=parent1.type)
    
    # Special handling for task chromosomes
    if parent1.type == "task":
        return _combine_task_chromosomes(parent1, parent2)
    else:
        return _combine_regular_chromosomes(parent1, parent2)

def _combine_task_chromosomes(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """Special combination logic for task chromosomes"""
    # For task chromosomes, prefer content closer to TARGET_LENGTH
    p1_value = max(0, TARGET_LENGTH - abs(len(parent1.content) - TARGET_LENGTH))
    p2_value = max(0, TARGET_LENGTH - abs(len(parent2.content) - TARGET_LENGTH))
    
    # If one parent is significantly better, bias toward it
    if p1_value > p2_value * 1.5:
        primary_weight = 0.7  # 70% chance to use parent1 content
    elif p2_value > p1_value * 1.5:
        primary_weight = 0.3  # 30% chance to use parent1 content
    else:
        primary_weight = 0.5  # Equal chance
    
    # Determine primary parent based on weighted probability
    if random.random() < primary_weight:
        primary_parent, secondary_parent = parent1, parent2
    else:
        primary_parent, secondary_parent = parent2, parent1
    
    # Find all hotspots in primary parent's content
    hotspots = [i for i, char in enumerate(primary_parent.content) if char in HOTSPOT_CHARS]
    if not hotspots:
        # If no hotspots, add some arbitrary points
        content_len = len(primary_parent.content)
        chunk_size = max(1, content_len // 5)
        hotspots = [i for i in range(chunk_size, content_len, chunk_size)]
    
    # Start with primary parent's content
    result = list(primary_parent.content)
    
    # Calculate target number of switches - aim for ~1 switch per chromosome
    target_switches = 1
    
    # Sort hotspots and select a subset for switching
    hotspots.sort()
    if hotspots:
        # Select random hotspots for switching
        switch_points = sorted(random.sample(hotspots, min(target_switches, len(hotspots))))
        
        # Perform the switches
        for point in switch_points:
            if random.random() < CHROMOSOME_SWITCH_PROBABILITY:
                # Take content from secondary parent from this point
                if point < len(secondary_parent.content):
                    secondary_content = list(secondary_parent.content[point:])
                    
                    # Adjust result length
                    if point + len(secondary_content) > len(result):
                        result = result[:point] + secondary_content
                    else:
                        result[point:point+len(secondary_content)] = secondary_content
    
    # Ensure we don't exceed MAX_CHARS
    combined_content = ''.join(result)
    if len(combined_content) > MAX_CHARS:
        combined_content = combined_content[:MAX_CHARS]
    
    # For task chromosomes, try to keep length close to TARGET_LENGTH
    if len(combined_content) > TARGET_LENGTH * 1.5:
        # Truncate to a reasonable length with some randomness
        max_length = int(TARGET_LENGTH * (1.0 + random.random() * 0.5))
        combined_content = combined_content[:max_length]
    
    return Chromosome(content=combined_content, type="task")

def _combine_regular_chromosomes(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """Combination logic for non-task chromosomes"""
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
        chunk_size = max(1, content_len // 5)
        hotspots = [i for i in range(chunk_size, content_len, chunk_size)]
    
    # Start with primary parent's content
    result = list(primary_parent.content)
    
    # Sort hotspots and select a subset for switching
    hotspots.sort()
    if hotspots:
        # Select 1-2 random hotspots for switching
        num_switches = random.randint(1, min(2, len(hotspots)))
        switch_points = sorted(random.sample(hotspots, num_switches))
        
        # Perform the switches
        for point in switch_points:
            if random.random() < CHROMOSOME_SWITCH_PROBABILITY:
                # Take content from secondary parent from this point
                if point < len(secondary_parent.content):
                    secondary_content = list(secondary_parent.content[point:])
                    
                    # Adjust result length
                    if point + len(secondary_content) > len(result):
                        result = result[:point] + secondary_content
                    else:
                        result[point:point+len(secondary_content)] = secondary_content
    
    # Ensure we don't exceed MAX_CHARS
    combined_content = ''.join(result)
    if len(combined_content) > MAX_CHARS:
        combined_content = combined_content[:MAX_CHARS]
    
    return Chromosome(content=combined_content, type=parent1.type)

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
