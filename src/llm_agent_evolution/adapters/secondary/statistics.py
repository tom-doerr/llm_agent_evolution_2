from typing import Dict, Any, List
import numpy as np
from collections import deque
import threading
from llm_agent_evolution.ports.secondary import StatisticsPort

class StatisticsAdapter(StatisticsPort):
    """Adapter for tracking and calculating statistics"""
    
    def __init__(self):
        """Initialize the statistics adapter"""
        self.rewards = []
        self.recent_rewards = deque(maxlen=100)  # Sliding window of last 100 evaluations
        self.best_reward = None
        self.worst_reward = None
        self.lock = threading.Lock()  # Thread safety
    
    def track_reward(self, reward: float) -> None:
        """Track a new reward value"""
        with self.lock:
            self.rewards.append(reward)
            self.recent_rewards.append(reward)
            
            # Update best and worst
            if self.best_reward is None or reward > self.best_reward:
                self.best_reward = reward
            if self.worst_reward is None or reward < self.worst_reward:
                self.worst_reward = reward
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.lock:
            if not self.rewards:
                return {
                    "count": 0,
                    "mean": None,
                    "median": None,
                    "std_dev": None,
                    "best": None,
                    "worst": None
                }
            
            return {
                "count": len(self.rewards),
                "mean": np.mean(self.rewards),
                "median": np.median(self.rewards),
                "std_dev": np.std(self.rewards),
                "best": self.best_reward,
                "worst": self.worst_reward
            }
    
    def get_sliding_window_stats(self, window_size: int = 100) -> Dict[str, Any]:
        """Get statistics for the sliding window of recent evaluations"""
        with self.lock:
            rewards_list = list(self.recent_rewards)
            if not rewards_list:
                return {
                    "window_size": window_size,
                    "count": 0,
                    "mean": None,
                    "median": None,
                    "std_dev": None
                }
            
            return {
                "window_size": window_size,
                "count": len(rewards_list),
                "mean": np.mean(rewards_list),
                "median": np.median(rewards_list),
                "std_dev": np.std(rewards_list)
            }
