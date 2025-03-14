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
    
    # Special handling for task chromosomes
    if parent1.type == "task":
        # Calculate how close each parent is to the target length
        p1_distance = abs(len(parent1.content) - TARGET_LENGTH)
        p2_distance = abs(len(parent2.content) - TARGET_LENGTH)
        
        # Bias toward the parent closer to target length
        if p1_distance < p2_distance:
            primary_parent, secondary_parent = parent1, parent2
        else:
            primary_parent, secondary_parent = parent2, parent1
    else:
        # For non-task chromosomes, randomly select primary parent
        if random.random() < 0.5:
            primary_parent, secondary_parent = parent1, parent2
        else:
            primary_parent, secondary_parent = parent2, parent1
    
    # Find hotspots for crossover
    hotspots = [i for i, char in enumerate(primary_parent.content) if char in HOTSPOT_CHARS]
    if not hotspots:
        # If no hotspots, add some arbitrary points
        content_len = len(primary_parent.content)
        chunk_size = max(1, content_len // 5)
        hotspots = [i for i in range(chunk_size, content_len, chunk_size)]
    
    # Start with primary parent's content
    result = list(primary_parent.content)
    
    # Sort hotspots
    hotspots.sort()
    if hotspots:
        # Select a random hotspot for switching
        switch_point = random.choice(hotspots)
        
        # Take content from secondary parent from this point
        if switch_point < len(secondary_parent.content):
            secondary_content = list(secondary_parent.content[switch_point:])
            
            # Adjust result length
            if switch_point + len(secondary_content) > len(result):
                result = result[:switch_point] + secondary_content
            else:
                result[switch_point:switch_point+len(secondary_content)] = secondary_content
    
    # Ensure we don't exceed MAX_CHARS
    combined_content = ''.join(result)
    if len(combined_content) > MAX_CHARS:
        combined_content = combined_content[:MAX_CHARS]
    
    # For task chromosomes, bias toward TARGET_LENGTH
    if parent1.type == "task" and len(combined_content) > TARGET_LENGTH:
        # High probability to truncate to exactly TARGET_LENGTH
        if random.random() < 0.8:
            combined_content = combined_content[:TARGET_LENGTH]
    
    return Chromosome(content=combined_content, type=parent1.type)

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
