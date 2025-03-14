import pytest
import sys
import os
import time

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_agent_evolution.adapters.secondary.statistics import StatisticsAdapter

def test_statistics_adapter_initialization():
    """Test creating a statistics adapter"""
    adapter = StatisticsAdapter()
    assert adapter is not None
    assert adapter.rewards == []
    assert adapter.best_reward is None
    assert adapter.worst_reward is None

def test_track_reward():
    """Test tracking rewards"""
    adapter = StatisticsAdapter()
    
    # Track some rewards
    adapter.track_reward(10)
    adapter.track_reward(20)
    adapter.track_reward(5)
    
    # Check basic stats
    assert len(adapter.rewards) == 3
    assert adapter.best_reward == 20
    assert adapter.worst_reward == 5
    
    # Check that recent_rewards works as a sliding window
    for i in range(100):
        adapter.track_reward(i)
    
    assert len(adapter.recent_rewards) == 100
    assert len(adapter.rewards) == 103

def test_get_stats():
    """Test getting statistics"""
    adapter = StatisticsAdapter()
    
    # Test with no rewards
    stats = adapter.get_stats()
    assert stats["count"] == 0
    assert stats["mean"] is None
    
    # Add some rewards
    adapter.track_reward(10)
    adapter.track_reward(20)
    adapter.track_reward(30)
    
    # Test with rewards
    stats = adapter.get_stats()
    assert stats["count"] == 3
    assert stats["mean"] == 20
    assert stats["median"] == 20
    assert stats["best"] == 30
    assert stats["worst"] == 10

def test_improvement_tracking():
    """Test tracking improvements over time"""
    adapter = StatisticsAdapter()
    
    # Add rewards with increasing values
    adapter.track_reward(10)
    time.sleep(0.1)  # Small delay to ensure different timestamps
    adapter.track_reward(20)
    time.sleep(0.1)
    adapter.track_reward(30)
    
    # Check improvement history
    history = adapter.get_improvement_history()
    assert len(history) == 2  # Two improvements
    
    # Check improvement values
    assert history[0]["improvement"] == 10  # 20 - 10
    assert history[1]["improvement"] == 10  # 30 - 20
    
    # Check that time_since_last is positive
    assert history[0]["time_since_last"] > 0
    assert history[1]["time_since_last"] > 0

def test_sliding_window_stats():
    """Test sliding window statistics"""
    adapter = StatisticsAdapter()
    
    # Add 150 rewards
    for i in range(150):
        adapter.track_reward(i)
    
    # Get sliding window stats (default window size is 100)
    window_stats = adapter.get_sliding_window_stats()
    
    # Should only have the last 100 rewards (50-149)
    assert window_stats["count"] == 100
    assert window_stats["mean"] == 99.5  # Average of 50-149
