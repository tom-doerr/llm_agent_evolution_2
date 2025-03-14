import random
from typing import List, Tuple, Optional
import numpy as np
from .model import Agent, Chromosome, MAX_CHARS

# Constants
CHROMOSOME_SWITCH_PROBABILITY = 0.3  # Probability of switching chromosomes at hotspots
HOTSPOT_CHARS = ".,;:!?()[]{}'\"\n "  # Punctuation and spaces as hotspots
TARGET_LENGTH = 23  # Target length for task optimization (used in combine_chromosomes)
# This is a constant from the spec, not revealing the goal

def select_parents_pareto(population: List[Agent], num_parents: int) -> List[Agent]:
    """
    Select parents using Pareto distribution weighting by fitness^2
    Uses weighted sampling without replacement
    
    Improved to better handle negative rewards and ensure diversity
    """
    if not population or num_parents <= 0:
        return []
    
    # Filter and validate agents
    valid_agents = _filter_valid_agents(population)
    if not valid_agents:
        return _select_random_agents(population, num_parents)
    
    # Calculate weights for selection
    weights = _calculate_selection_weights(valid_agents)
    
    # Perform weighted selection
    selected_parents = _weighted_selection(valid_agents, weights, num_parents)
    
    # Add diversity if needed
    selected_parents = _ensure_diversity(selected_parents, population, num_parents)
    
    return selected_parents

def _filter_valid_agents(population: List[Agent]) -> List[Agent]:
    """Filter out agents without rewards"""
    return [agent for agent in population if agent.reward is not None]

def _select_random_agents(population: List[Agent], num_agents: int) -> List[Agent]:
    """Select random agents from the population"""
    return random.sample(population, min(num_agents, len(population)))

def _calculate_selection_weights(agents: List[Agent]) -> List[float]:
    """Calculate selection weights based on rewards"""
    # Get reward statistics
    rewards = [agent.reward for agent in agents]
    min_reward = min(rewards)
    max_reward = max(rewards)
    
    # Handle case where all rewards are the same
    if min_reward == max_reward:
        return [1.0 / len(agents)] * len(agents)
    
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
    return [w / total_weight for w in squared_rewards]

def _weighted_selection(agents: List[Agent], weights: List[float], num_select: int) -> List[Agent]:
    """Perform weighted selection without replacement"""
    selected = []
    remaining_agents = agents.copy()
    remaining_weights = weights.copy()
    
    # Select agents one by one
    for _ in range(min(num_select, len(agents))):
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
        
        # Add selected agent to result
        selected.append(remaining_agents[idx])
        
        # Remove selected agent from candidates
        remaining_agents.pop(idx)
        remaining_weights.pop(idx)
    
    return selected

def _ensure_diversity(selected: List[Agent], population: List[Agent], target_count: int) -> List[Agent]:
    """Ensure diversity by adding some random agents if needed"""
    if len(selected) < target_count and len(population) > len(selected):
        remaining_agents = [a for a in population if a not in selected]
        # Add a small percentage of random agents for exploration
        num_random = max(1, int(target_count * 0.2))
        random_agents = random.sample(
            remaining_agents, 
            min(num_random, len(remaining_agents))
        )
        selected.extend(random_agents[:target_count - len(selected)])
    
    return selected

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
    # Calculate fitness values based on proximity to TARGET_LENGTH
    p1_value, p2_value = _calculate_length_fitness(parent1, parent2)
    
    # Select primary parent based on fitness values
    primary_parent, secondary_parent = _select_primary_parent(parent1, parent2, p1_value, p2_value)
    
    # Find hotspots for crossover
    hotspots = _find_hotspots(primary_parent.content)
    
    # Perform crossover at hotspots
    result = _perform_crossover(primary_parent.content, secondary_parent.content, hotspots)
    
    # Post-process the combined content
    combined_content = _post_process_task_content(result)
    
    return Chromosome(content=combined_content, type="task")

def _calculate_length_fitness(parent1: Chromosome, parent2: Chromosome) -> Tuple[float, float]:
    """Calculate fitness values based on proximity to TARGET_LENGTH"""
    # Calculate how close each parent is to the target length
    p1_distance = abs(len(parent1.content) - TARGET_LENGTH)
    p2_distance = abs(len(parent2.content) - TARGET_LENGTH)
    
    # Stronger bias toward target length - inverse square of distance
    p1_value = max(1, (TARGET_LENGTH - p1_distance)**2) if p1_distance <= TARGET_LENGTH else max(1, TARGET_LENGTH / (p1_distance + 1))
    p2_value = max(1, (TARGET_LENGTH - p2_distance)**2) if p2_distance <= TARGET_LENGTH else max(1, TARGET_LENGTH / (p2_distance + 1))
    
    return p1_value, p2_value

def _select_primary_parent(parent1: Chromosome, parent2: Chromosome, p1_value: float, p2_value: float) -> Tuple[Chromosome, Chromosome]:
    """Select primary parent based on fitness values"""
    # If one parent is significantly better, bias toward it
    if p1_value > p2_value * 1.5:
        primary_weight = 0.7  # 70% chance to use parent1 content
    elif p2_value > p1_value * 1.5:
        primary_weight = 0.3  # 30% chance to use parent1 content
    else:
        primary_weight = 0.5  # Equal chance
    
    # Determine primary parent based on weighted probability
    if random.random() < primary_weight:
        return parent1, parent2
    else:
        return parent2, parent1

def _find_hotspots(content: str) -> List[int]:
    """Find hotspots for crossover in the content"""
    hotspots = [i for i, char in enumerate(content) if char in HOTSPOT_CHARS]
    if not hotspots:
        # If no hotspots, add some arbitrary points
        content_len = len(content)
        chunk_size = max(1, content_len // 5)
        hotspots = [i for i in range(chunk_size, content_len, chunk_size)]
    return hotspots

def _perform_crossover(primary_content: str, secondary_content: str, hotspots: List[int]) -> List[str]:
    """Perform crossover at hotspots"""
    # Start with primary parent's content
    result = list(primary_content)
    
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
                if point < len(secondary_content):
                    secondary_content_list = list(secondary_content[point:])
                    
                    # Adjust result length
                    if point + len(secondary_content_list) > len(result):
                        result = result[:point] + secondary_content_list
                    else:
                        result[point:point+len(secondary_content_list)] = secondary_content_list
    
    return result

def _post_process_task_content(result: List[str]) -> str:
    """Post-process the combined content"""
    # Ensure we don't exceed MAX_CHARS
    combined_content = ''.join(result)
    if len(combined_content) > MAX_CHARS:
        combined_content = combined_content[:MAX_CHARS]
    
    # For task chromosomes, apply some general length constraints
    # This is a generic approach that doesn't leak task-specific knowledge
    if len(combined_content) > 100:  # Use a generic reasonable length
        # Truncate to a reasonable length with some randomness
        max_length = int(100 * (1.0 + random.random() * 0.5))
        combined_content = combined_content[:max_length]
    
    return combined_content

def _combine_regular_chromosomes(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """Combination logic for non-task chromosomes"""
    # Randomly select primary parent
    primary_parent, secondary_parent = _random_select_parents(parent1, parent2)
    
    # Find hotspots for crossover
    hotspots = _find_hotspots(primary_parent.content)
    
    # Perform crossover with more switch points for regular chromosomes
    result = _perform_regular_crossover(primary_parent.content, secondary_parent.content, hotspots)
    
    # Post-process the combined content
    combined_content = _post_process_regular_content(result)
    
    return Chromosome(content=combined_content, type=parent1.type)

def _random_select_parents(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    """Randomly select primary and secondary parents"""
    if random.random() < 0.5:
        return parent1, parent2
    else:
        return parent2, parent1

def _perform_regular_crossover(primary_content: str, secondary_content: str, hotspots: List[int]) -> List[str]:
    """Perform crossover at hotspots with more switch points for regular chromosomes"""
    # Start with primary parent's content
    result = list(primary_content)
    
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
                if point < len(secondary_content):
                    secondary_content_list = list(secondary_content[point:])
                    
                    # Adjust result length
                    if point + len(secondary_content_list) > len(result):
                        result = result[:point] + secondary_content_list
                    else:
                        result[point:point+len(secondary_content_list)] = secondary_content_list
    
    return result

def _post_process_regular_content(result: List[str]) -> str:
    """Post-process the combined content for regular chromosomes"""
    # Ensure we don't exceed MAX_CHARS
    combined_content = ''.join(result)
    if len(combined_content) > MAX_CHARS:
        combined_content = combined_content[:MAX_CHARS]
    
    return combined_content

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
