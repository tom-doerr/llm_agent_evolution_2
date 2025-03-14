import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from matplotlib.ticker import MaxNLocator

class VisualizationAdapter:
    """Adapter for creating visualizations of evolution progress"""
    
    def __init__(self, output_dir: str = "visualizations"):
        """Initialize the visualization adapter"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = int(time.time())
    
    def plot_rewards_over_time(self, rewards: List[float], window_size: int = 10) -> str:
        """Plot rewards over time with a moving average"""
        if not rewards:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot raw rewards
        ax.plot(rewards, 'o', alpha=0.3, color='lightgray', label='Individual rewards')
        
        # Calculate and plot moving average if we have enough data
        if len(rewards) >= window_size:
            moving_avg = []
            for i in range(len(rewards) - window_size + 1):
                moving_avg.append(np.mean(rewards[i:i+window_size]))
            
            # Plot moving average at the correct x positions
            ax.plot(range(window_size-1, len(rewards)), moving_avg, 
                   linewidth=2, color='blue', label=f'Moving average (window={window_size})')
        
        # Add best reward line
        best_reward = max(rewards)
        ax.axhline(y=best_reward, color='green', linestyle='--', 
                  label=f'Best reward: {best_reward:.2f}')
        
        # Add labels and title
        ax.set_xlabel('Evaluation Number')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Evolution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Ensure x-axis uses integers
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Save the figure
        filename = f"{self.output_dir}/rewards_{self.timestamp}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        
        return filename
    
    def plot_reward_distribution(self, rewards: List[float]) -> str:
        """Plot histogram of reward distribution"""
        if not rewards:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        ax.hist(rewards, bins=20, alpha=0.7, color='blue')
        
        # Add mean and median lines
        mean_reward = np.mean(rewards)
        median_reward = np.median(rewards)
        
        ax.axvline(x=mean_reward, color='red', linestyle='--', 
                  label=f'Mean: {mean_reward:.2f}')
        ax.axvline(x=median_reward, color='green', linestyle='--', 
                  label=f'Median: {median_reward:.2f}')
        
        # Add labels and title
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save the figure
        filename = f"{self.output_dir}/distribution_{self.timestamp}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        
        return filename
    
    def plot_top_agents_comparison(self, agents_data: List[Dict[str, Any]], 
                                  metric: str = 'reward', top_n: int = 10) -> str:
        """Plot comparison of top agents by specified metric"""
        if not agents_data:
            return None
            
        # Sort agents by the metric and take top N
        sorted_agents = sorted(agents_data, key=lambda x: x.get(metric, 0), reverse=True)[:top_n]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract data for plotting
        ids = [agent.get('id', f'Agent {i}')[:8] + '...' for i, agent in enumerate(sorted_agents)]
        values = [agent.get(metric, 0) for agent in sorted_agents]
        
        # Create bar chart
        bars = ax.bar(ids, values, color='skyblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom')
        
        # Add labels and title
        ax.set_xlabel('Agent ID')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Top {top_n} Agents by {metric.capitalize()}')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Save the figure
        filename = f"{self.output_dir}/top_agents_{self.timestamp}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        
        return filename
    
    def plot_improvement_rate(self, improvement_history: List[Dict[str, Any]]) -> str:
        """Plot improvement rate over time"""
        if not improvement_history:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        timestamps = [entry['timestamp'] for entry in improvement_history]
        improvements = [entry['improvement'] for entry in improvement_history]
        
        # Convert timestamps to relative time in minutes
        start_time = min(timestamps)
        relative_times = [(t - start_time) / 60 for t in timestamps]
        
        # Plot improvements
        ax.plot(relative_times, improvements, 'o-', color='green', linewidth=2)
        
        # Calculate and plot moving average if we have enough data
        window_size = min(5, len(improvements))
        if len(improvements) >= window_size:
            moving_avg = []
            for i in range(len(improvements) - window_size + 1):
                moving_avg.append(np.mean(improvements[i:i+window_size]))
            
            # Plot moving average at the correct x positions
            ax.plot(relative_times[window_size-1:], moving_avg, 
                   linewidth=2, color='blue', label=f'Moving average (window={window_size})')
        
        # Add labels and title
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Improvement')
        ax.set_title('Improvement Rate Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Save the figure
        filename = f"{self.output_dir}/improvement_rate_{self.timestamp}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        
        return filename
    
    def create_evolution_dashboard(self, stats: Dict[str, Any], 
                                  rewards_history: List[float],
                                  top_agents: List[Dict[str, Any]],
                                  improvement_history: List[Dict[str, Any]] = None) -> List[str]:
        """Create a comprehensive dashboard with multiple visualizations"""
        filenames = []
        
        # Plot rewards over time
        rewards_plot = self.plot_rewards_over_time(rewards_history)
        if rewards_plot:
            filenames.append(rewards_plot)
        
        # Plot reward distribution
        dist_plot = self.plot_reward_distribution(rewards_history)
        if dist_plot:
            filenames.append(dist_plot)
        
        # Plot top agents comparison
        if top_agents:
            top_plot = self.plot_top_agents_comparison(top_agents)
            if top_plot:
                filenames.append(top_plot)
        
        # Plot improvement rate
        if improvement_history:
            improvement_plot = self.plot_improvement_rate(improvement_history)
            if improvement_plot:
                filenames.append(improvement_plot)
        
        return filenames
