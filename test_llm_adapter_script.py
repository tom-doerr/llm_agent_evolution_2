#!/usr/bin/env python3
"""
Script to test the LLM adapter directly
"""
import sys
import os
import time

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from llm_agent_evolution.domain.model import Chromosome, Agent
from llm_agent_evolution.adapters.secondary.llm import DSPyLLMAdapter

def test_llm_adapter():
    """Test the LLM adapter directly"""
    print("Testing LLM adapter...")
    
    # Create LLM adapter
    llm = DSPyLLMAdapter(model_name="openrouter/google/gemini-2.0-flash-001")
    
    # Test mutation
    print("\nTesting mutation...")
    chromosome = Chromosome(content="This is a test content", type="task")
    mutation_instructions = "Add more a's to this content"
    
    start_time = time.time()
    result = llm.generate_mutation(chromosome, mutation_instructions)
    duration = time.time() - start_time
    
    print(f"Original content: {chromosome.content}")
    print(f"Mutation instructions: {mutation_instructions}")
    print(f"Result content: {result.content}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Test mate selection
    print("\nTesting mate selection...")
    agent1 = Agent(
        task_chromosome=Chromosome(content="Agent 1 DNA", type="task"),
        mate_selection_chromosome=Chromosome(content="Select the best mate", type="mate_selection"),
        mutation_chromosome=Chromosome(content="Mutate wisely", type="mutation")
    )
    
    agent2 = Agent(
        task_chromosome=Chromosome(content="Agent 2 DNA with some a's: aaaaa", type="task"),
        mate_selection_chromosome=Chromosome(content="", type="mate_selection"),
        mutation_chromosome=Chromosome(content="", type="mutation")
    )
    
    agent3 = Agent(
        task_chromosome=Chromosome(content="Agent 3 DNA with more a's: aaaaaaaaaa", type="task"),
        mate_selection_chromosome=Chromosome(content="", type="mate_selection"),
        mutation_chromosome=Chromosome(content="", type="mutation")
    )
    
    start_time = time.time()
    selected = llm.select_mate(agent1, [agent2, agent3])
    duration = time.time() - start_time
    
    print(f"Agent 1 DNA: {agent1.task_chromosome.content}")
    print(f"Agent 2 DNA: {agent2.task_chromosome.content}")
    print(f"Agent 3 DNA: {agent3.task_chromosome.content}")
    print(f"Selected agent DNA: {selected.task_chromosome.content}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Test evaluation
    print("\nTesting evaluation...")
    outputs = [
        "",
        "a",
        "aaa",
        "aaaaaaaa",
        "aaaaaaaaaaaaaaaaaaaaaaa",  # 23 a's (optimal)
        "aaaaaaaaaaaaaaaaaaaaaaaa",  # 24 a's (1 too many)
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # 29 a's (6 too many)
    ]
    
    for output in outputs:
        reward = llm.evaluate_task_output(output)
        print(f"Output: '{output}', Reward: {reward}")

if __name__ == "__main__":
    test_llm_adapter()
