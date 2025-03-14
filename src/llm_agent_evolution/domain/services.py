import random
from typing import List, Tuple
from .model import Agent, Chromosome, MAX_CHARS, TARGET_LENGTH

# Constants
CHROMOSOME_SWITCH_PROBABILITY = 0.3  # Probability of switching chromosomes at hotspots
HOTSPOT_CHARS = ".,;:!?()[]{}'\"\n "  # Punctuation and spaces as hotspots

def select_parents_pareto(population: List[Agent], num_parents: int) -> List[Agent]:
    """Select parents using Pareto distribution weighting by fitness^2"""
    if not population or num_parents <= 0:
        return []
    
    # Filter out agents without rewards
    valid_agents = [agent for agent in population if agent.reward is not None]
    if not valid_agents:
        return random.sample(population, min(num_parents, len(population)))
    
    # Get reward statistics
    rewards = [agent.reward for agent in valid_agents]
    min_reward = min(rewards)
    
    # Shift rewards to positive range if needed
    if min_reward < 0:
        shift = abs(min_reward) + 1.0
        adjusted_rewards = [(r + shift) for r in rewards]
    else:
        adjusted_rewards = rewards.copy()
    
    # Square the rewards for Pareto distribution (emphasize higher rewards)
    squared_rewards = [r**2 for r in adjusted_rewards]
    
    # Normalize weights
    total_weight = sum(squared_rewards)
    weights = [w / total_weight for w in squared_rewards]
    
    # Weighted sampling without replacement
    selected = []
    remaining_agents = valid_agents.copy()
    remaining_weights = weights.copy()
    
    for _ in range(min(num_parents, len(valid_agents))):
        if not remaining_agents:
            break
            
        # Normalize remaining weights
        weight_sum = sum(remaining_weights)
        if weight_sum <= 0:
            idx = random.randrange(len(remaining_agents))
        else:
            normalized_weights = [w / weight_sum for w in remaining_weights]
            idx = random.choices(range(len(remaining_agents)), weights=normalized_weights, k=1)[0]
        
        selected.append(remaining_agents[idx])
        remaining_agents.pop(idx)
        remaining_weights.pop(idx)
    
    # Add diversity if needed
    if len(selected) < num_parents and len(population) > len(selected):
        remaining_agents = [a for a in population if a not in selected]
        random_agents = random.sample(
            remaining_agents, 
            min(num_parents - len(selected), len(remaining_agents))
        )
        selected.extend(random_agents)
    
    return selected

def combine_chromosomes(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """Combine two chromosomes by switching at hotspots"""
    # Handle empty content cases
    if not parent1.content and not parent2.content:
        return Chromosome(content="", type=parent1.type)
    
    if not parent1.content:
        return Chromosome(content=parent2.content, type=parent1.type)
    
    if not parent2.content:
        return Chromosome(content=parent1.content, type=parent1.type)
    
    # Select primary parent based on type and target length
    primary_parent, secondary_parent = _select_primary_parent(parent1, parent2)
    
    # Find crossover points
    hotspots = _find_hotspots(primary_parent.content)
    
    # Perform crossover
    combined_content = _perform_crossover(primary_parent.content, secondary_parent.content, hotspots)
    
    # Apply length constraints
    if len(combined_content) > MAX_CHARS:
        combined_content = combined_content[:MAX_CHARS]
    
    # For task chromosomes, bias toward TARGET_LENGTH
    if parent1.type == "task" and len(combined_content) > TARGET_LENGTH:
        if random.random() < 0.8:
            combined_content = combined_content[:TARGET_LENGTH]
    
    return Chromosome(content=combined_content, type=parent1.type)

def _select_primary_parent(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    """Select which parent should be primary based on chromosome type and content"""
    if parent1.type == "task":
        # For task chromosomes, prefer the one closer to target length
        p1_distance = abs(len(parent1.content) - TARGET_LENGTH)
        p2_distance = abs(len(parent2.content) - TARGET_LENGTH)
        
        return (parent1, parent2) if p1_distance < p2_distance else (parent2, parent1)
    else:
        # For other chromosomes, random selection
        return (parent1, parent2) if random.random() < 0.5 else (parent2, parent1)

def _find_hotspots(content: str) -> List[int]:
    """Find suitable crossover points in the content"""
    hotspots = [i for i, char in enumerate(content) if char in HOTSPOT_CHARS]
    
    if not hotspots:
        # If no hotspots, create artificial ones
        content_len = len(content)
        chunk_size = max(1, content_len // 5)
        hotspots = [i for i in range(chunk_size, content_len, chunk_size)]
    
    return sorted(hotspots)

def _perform_crossover(primary_content: str, secondary_content: str, hotspots: List[int]) -> str:
    """Perform the actual crossover operation at selected hotspots"""
    if not hotspots:
        return primary_content
        
    # Start with primary parent's content
    result = list(primary_content)
    
    # Ensure we have at least one chromosome jump on average
    # Select 1-2 random hotspots for switching
    num_switches = random.randint(1, min(2, len(hotspots)))
    switch_points = sorted(random.sample(hotspots, num_switches))
    
    # Perform the switches
    for switch_point in switch_points:
        # Take content from secondary parent from this point
        if switch_point < len(secondary_content):
            secondary_part = list(secondary_content[switch_point:])
            
            # Adjust result length
            if switch_point + len(secondary_part) > len(result):
                result = result[:switch_point] + secondary_part
            else:
                result[switch_point:switch_point+len(secondary_part)] = secondary_part
    
    return ''.join(result)

def mate_agents(parent1: Agent, parent2: Agent) -> Agent:
    """Create a new agent by combining chromosomes from two parents"""
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
