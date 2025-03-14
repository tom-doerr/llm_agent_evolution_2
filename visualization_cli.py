#!/usr/bin/env python3
"""
CLI tool for generating visualizations from evolution logs and data
"""
import sys
import os
import argparse
import glob
import json
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from llm_agent_evolution.adapters.secondary.visualization import VisualizationAdapter

def parse_log_file(log_file):
    """Parse the log file to extract rewards and agent data"""
    rewards = []
    agents = []
    
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found")
        return rewards, agents
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "EVALUATION" in line:
            agent_id = None
            reward = None
            task_content = None
            
            # Look for agent details in the next few lines
            for j in range(i+1, min(i+10, len(lines))):
                if "Agent ID:" in lines[j]:
                    agent_id = lines[j].split("Agent ID:")[1].strip()
                elif "Reward:" in lines[j]:
                    try:
                        reward = float(lines[j].split("Reward:")[1].strip())
                        rewards.append(reward)
                    except ValueError:
                        pass
                elif "Task Chromosome" in lines[j]:
                    task_content = lines[j].split(":")[-1].strip()
            
            if agent_id and reward is not None:
                agents.append({
                    "id": agent_id,
                    "reward": reward,
                    "task_length": len(task_content) if task_content else 0
                })
        
        i += 1
    
    return rewards, agents

def main():
    """Main entry point for the visualization CLI"""
    parser = argparse.ArgumentParser(description="Generate visualizations from evolution data")
    parser.add_argument("--log-file", default="evolution.log", help="Log file to analyze")
    parser.add_argument("--output-dir", default="visualizations", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Create visualization adapter
    viz = VisualizationAdapter(output_dir=args.output_dir)
    
    # Parse log file
    print(f"Parsing log file: {args.log_file}")
    rewards, agents = parse_log_file(args.log_file)
    
    if not rewards:
        print("No reward data found in log file")
        return 1
    
    print(f"Found {len(rewards)} reward entries and {len(agents)} agent entries")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Rewards over time
    rewards_file = viz.plot_rewards_over_time(rewards)
    if rewards_file:
        print(f"Rewards plot saved to: {rewards_file}")
    
    # Reward distribution
    dist_file = viz.plot_reward_distribution(rewards)
    if dist_file:
        print(f"Distribution plot saved to: {dist_file}")
    
    # Top agents comparison
    if agents:
        top_file = viz.plot_top_agents_comparison(agents)
        if top_file:
            print(f"Top agents plot saved to: {top_file}")
    
    print(f"\nAll visualizations saved to: {os.path.abspath(args.output_dir)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
