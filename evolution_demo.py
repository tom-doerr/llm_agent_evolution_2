#!/usr/bin/env python3
"""
Detailed demonstration of the evolution process showing each step
"""
import sys
import os
import time
import argparse
from typing import List, Dict, Any, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from llm_agent_evolution.domain.model import Agent, Chromosome
from llm_agent_evolution.domain.services import select_parents_pareto, mate_agents
from llm_agent_evolution.adapters.secondary.llm import DSPyLLMAdapter
from llm_agent_evolution.adapters.secondary.mock_llm import MockLLMAdapter

def print_separator(title: str = None):
    """Print a separator line with optional title"""
    width = 80
    if title:
        print("\n" + "=" * 10 + f" {title} " + "=" * (width - len(title) - 12) + "\n")
    else:
        print("\n" + "=" * width + "\n")

def print_agent(agent: Agent, detailed: bool = False):
    """Print agent information"""
    print(f"Agent ID: {agent.id}")
    print(f"Reward: {agent.reward}")
    print(f"Task Chromosome: '{agent.task_chromosome.content[:50]}{'...' if len(agent.task_chromosome.content) > 50 else ''}'")
    
    if detailed:
        print("\nMate Selection Chromosome:")
        print(f"{agent.mate_selection_chromosome.content}")
        
        print("\nMutation Chromosome:")
        print(f"{agent.mutation_chromosome.content}")

def run_evolution_demo():
    """Run a detailed demonstration of the evolution process"""
    parser = argparse.ArgumentParser(description="Evolution Process Demonstration")
    parser.add_argument("--use-mock", action="store_true", help="Use mock LLM instead of real LLM")
    parser.add_argument("--initial-content", type=str, default="a", help="Initial content for task chromosome")
    args = parser.parse_args()
    
    # Create LLM adapter
    if args.use_mock:
        print("Using mock LLM for demonstration")
        llm_adapter = MockLLMAdapter(seed=42)
    else:
        print("Using real LLM for demonstration")
        llm_adapter = DSPyLLMAdapter(model_name="openrouter/google/gemini-2.0-flash-001")
    
    # Initialize population
    print_separator("INITIALIZATION")
    print("Creating initial population with 5 agents...")
    
    population = []
    for i in range(5):
        # Initial task chromosome with some content
        task_content = args.initial_content
        
        # Initial mate selection chromosome with instructions
        mate_selection_content = """
        Select the mate with the highest reward.
        If rewards are equal, choose the one with more 'a' characters.
        """
        
        # Initial mutation chromosome with instructions
        mutation_content = """
        Rephrase the content to include more 'a' characters.
        Keep the length around 23 characters.
        Try different patterns and placements of 'a' characters.
        """
        
        agent = Agent(
            task_chromosome=Chromosome(content=task_content, type="task"),
            mate_selection_chromosome=Chromosome(content=mate_selection_content, type="mate_selection"),
            mutation_chromosome=Chromosome(content=mutation_content, type="mutation")
        )
        population.append(agent)
        
        print(f"\nCreated Agent {i+1}:")
        print_agent(agent, detailed=True)
    
    # Evaluate initial population
    print_separator("INITIAL EVALUATION")
    print("Evaluating initial population...")
    
    for agent in population:
        # Evaluate using the hidden goal (a's in first 23 chars, penalty for exceeding)
        output = agent.task_chromosome.content
        a_count = output[:23].count('a')
        length_penalty = max(0, len(output) - 23)
        reward = a_count - length_penalty
        
        agent.reward = reward
        
        print(f"\nAgent {agent.id} evaluation:")
        print(f"Content: '{output}'")
        print(f"a_count: {a_count}, length_penalty: {length_penalty}")
        print(f"Reward: {reward}")
    
    # Evolution loop
    steps_completed = 0
    
    while True:
        print_separator(f"EVOLUTION STEP {steps_completed + 1}")
        
        # 1. Parent Selection
        print("1. PARENT SELECTION")
        print("Selecting parents using Pareto distribution...")
        
        parents = select_parents_pareto(population, 2)
        
        print("\nSelected parents:")
        for i, parent in enumerate(parents):
            print(f"\nParent {i+1}:")
            print_agent(parent)
        
        # 2. Mate Selection
        print("\n2. MATE SELECTION")
        print("Using first parent's mate selection chromosome to choose a mate...")
        
        parent1 = parents[0]
        parent2 = llm_adapter.select_mate(parent1, [p for p in parents[1:]])
        
        print("\nSelected mate:")
        print_agent(parent2)
        
        # 3. Mating
        print("\n3. MATING")
        print("Creating new agent by combining chromosomes from both parents...")
        
        new_agent = mate_agents(parent1, parent2)
        
        print("\nNew agent after mating:")
        print_agent(new_agent, detailed=True)
        
        # 4. Mutation
        print("\n4. MUTATION")
        print("Mutating the new agent using its mutation chromosome...")
        
        print("\nMutation instructions:")
        print(new_agent.mutation_chromosome.content)
        
        # Mutate task chromosome
        print("\nMutating task chromosome...")
        print(f"Before: '{new_agent.task_chromosome.content}'")
        
        task_chromosome = llm_adapter.generate_mutation(
            new_agent.task_chromosome,
            new_agent.mutation_chromosome.content
        )
        
        print(f"After: '{task_chromosome.content}'")
        
        # Mutate mate selection chromosome
        print("\nMutating mate selection chromosome...")
        print(f"Before: '{new_agent.mate_selection_chromosome.content}'")
        
        mate_selection_chromosome = llm_adapter.generate_mutation(
            new_agent.mate_selection_chromosome,
            new_agent.mutation_chromosome.content
        )
        
        print(f"After: '{mate_selection_chromosome.content}'")
        
        # Mutate mutation chromosome
        print("\nMutating mutation chromosome...")
        print(f"Before: '{new_agent.mutation_chromosome.content}'")
        
        mutation_chromosome = llm_adapter.generate_mutation(
            new_agent.mutation_chromosome,
            new_agent.mutation_chromosome.content
        )
        
        print(f"After: '{mutation_chromosome.content}'")
        
        # Create mutated agent
        mutated_agent = Agent(
            task_chromosome=task_chromosome,
            mate_selection_chromosome=mate_selection_chromosome,
            mutation_chromosome=mutation_chromosome
        )
        
        # 5. Evaluation
        print("\n5. EVALUATION")
        print("Evaluating the mutated agent...")
        
        output = mutated_agent.task_chromosome.content
        a_count = output[:23].count('a')
        length_penalty = max(0, len(output) - 23)
        reward = a_count - length_penalty
        
        mutated_agent.reward = reward
        
        print(f"Content: '{output}'")
        print(f"a_count: {a_count}, length_penalty: {length_penalty}")
        print(f"Reward: {reward}")
        
        # 6. Population Update
        print("\n6. POPULATION UPDATE")
        print("Adding the mutated agent to the population...")
        
        population.append(mutated_agent)
        
        # Sort population by reward and keep only the top agents
        population = sorted(
            population,
            key=lambda a: a.reward if a.reward is not None else float('-inf'),
            reverse=True
        )[:5]  # Keep only top 5 agents
        
        print("\nUpdated population:")
        for i, agent in enumerate(population):
            print(f"\nAgent {i+1} (ID: {agent.id}):")
            print(f"Reward: {agent.reward}")
            print(f"Task: '{agent.task_chromosome.content}'")
        
        # Increment step counter
        steps_completed += 1
        
        # Ask user if they want to continue
        print_separator("CONTINUE?")
        print(f"Completed {steps_completed} evolution steps.")
        
        while True:
            try:
                steps = input("Enter number of steps to run (0 to exit): ")
                steps = int(steps)
                break
            except ValueError:
                print("Please enter a valid number.")
        
        if steps <= 0:
            break
        
        # Run the specified number of steps without detailed output
        for _ in range(steps - 1):
            # Quick evolution step without detailed output
            parents = select_parents_pareto(population, 2)
            parent1 = parents[0]
            parent2 = llm_adapter.select_mate(parent1, [p for p in parents[1:]])
            new_agent = mate_agents(parent1, parent2)
            
            # Mutate
            task_chromosome = llm_adapter.generate_mutation(
                new_agent.task_chromosome,
                new_agent.mutation_chromosome.content
            )
            
            mate_selection_chromosome = llm_adapter.generate_mutation(
                new_agent.mate_selection_chromosome,
                new_agent.mutation_chromosome.content
            )
            
            mutation_chromosome = llm_adapter.generate_mutation(
                new_agent.mutation_chromosome,
                new_agent.mutation_chromosome.content
            )
            
            mutated_agent = Agent(
                task_chromosome=task_chromosome,
                mate_selection_chromosome=mate_selection_chromosome,
                mutation_chromosome=mutation_chromosome
            )
            
            # Evaluate
            output = mutated_agent.task_chromosome.content
            a_count = output[:23].count('a')
            length_penalty = max(0, len(output) - 23)
            reward = a_count - length_penalty
            
            mutated_agent.reward = reward
            
            # Update population
            population.append(mutated_agent)
            population = sorted(
                population,
                key=lambda a: a.reward if a.reward is not None else float('-inf'),
                reverse=True
            )[:5]  # Keep only top 5 agents
            
            steps_completed += 1
            print(f"Completed step {steps_completed} (silent mode)")
    
    # Final results
    print_separator("FINAL RESULTS")
    print(f"Evolution completed after {steps_completed} steps.")
    
    print("\nFinal population:")
    for i, agent in enumerate(population):
        print(f"\nAgent {i+1} (ID: {agent.id}):")
        print_agent(agent, detailed=True)
    
    print("\nBest agent:")
    best_agent = population[0]
    print_agent(best_agent, detailed=True)

if __name__ == "__main__":
    run_evolution_demo()
