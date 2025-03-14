from typing import Dict, Any, List
import numpy as np
from collections import deque
import threading
import time
from llm_agent_evolution.ports.secondary import StatisticsPort

class StatisticsAdapter(StatisticsPort):
    """Adapter for tracking and calculating statistics"""
    
    def __init__(self):
        """Initialize the statistics adapter"""
        self.rewards = []
        self.recent_rewards = deque(maxlen=100)  # Sliding window of last 100 evaluations
        self.best_reward = None
        self.worst_reward = None
        self.reward_history = []  # Track rewards over time with timestamps
        self.improvement_rate = []  # Track improvement rate over time
        self.last_best_reward = None
        self.last_best_time = time.time()
        self.lock = threading.Lock()  # Thread safety
    
    def track_reward(self, reward: float) -> None:
        """Track a new reward value"""
        current_time = time.time()
        
        with self.lock:
            self.rewards.append(reward)
            self.recent_rewards.append(reward)
            
            # Track reward with timestamp
            self.reward_history.append((current_time, reward))
            
            # Update best and worst
            if self.best_reward is None or reward > self.best_reward:
                # If this is a new best reward, track improvement
                if self.best_reward is not None:
                    time_since_last_best = current_time - self.last_best_time
                    improvement = reward - self.best_reward
                    self.improvement_rate.append((current_time, improvement, time_since_last_best))
                
                self.best_reward = reward
                self.last_best_time = current_time
                
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
                    "worst": None,
                    "improvement_rate": None,
                    "time_since_last_best": None
                }
            
            # Calculate improvement rate (improvements per minute)
            current_time = time.time()
            time_since_last_best = current_time - self.last_best_time
            
            # Calculate improvement rate over the last 10 improvements or all if fewer
            recent_improvements = self.improvement_rate[-10:] if self.improvement_rate else []
            if recent_improvements:
                total_improvement = sum(imp for _, imp, _ in recent_improvements)
                total_time = sum(t for _, _, t in recent_improvements)
                improvement_per_minute = (total_improvement / total_time * 60) if total_time > 0 else 0
            else:
                improvement_per_minute = 0
            
            return {
                "count": len(self.rewards),
                "mean": np.mean(self.rewards),
                "median": np.median(self.rewards),
                "std_dev": np.std(self.rewards),
                "best": self.best_reward,
                "worst": self.worst_reward,
                "improvement_rate": improvement_per_minute,
                "time_since_last_best": time_since_last_best
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
    
    def get_improvement_history(self) -> List[Dict[str, Any]]:
        """Get the history of improvements for visualization"""
        with self.lock:
            if not self.improvement_rate:
                return []
            
            return [
                {
                    "timestamp": timestamp,
                    "improvement": improvement,
                    "time_since_last": time_since_last
                }
                for timestamp, improvement, time_since_last in self.improvement_rate
            ]
