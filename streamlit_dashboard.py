#!/usr/bin/env python3
"""
Streamlit dashboard for LLM Agent Evolution
Provides live insights into the evolution process
"""
import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import json

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Set page configuration
st.set_page_config(
    page_title="LLM Agent Evolution Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables for tracking state
last_file_size = 0
last_update_time = time.time()
evolution_data = {
    "rewards": [],
    "agents": [],
    "stats": {
        "population_size": 0,
        "total_evaluations": 0,
        "mean_reward": 0,
        "median_reward": 0,
        "std_dev": 0,
        "best_reward": 0,
        "worst_reward": 0
    },
    "window_stats": {
        "count": 0,
        "mean": 0,
        "median": 0,
        "std_dev": 0
    },
    "best_agent": {
        "id": "",
        "reward": 0,
        "task_content": ""
    }
}

def parse_log_file(log_file):
    """Parse the log file to extract rewards, agents, and statistics"""
    global last_file_size, evolution_data
    
    if not os.path.exists(log_file):
        return False
    
    current_size = os.path.getsize(log_file)
    if current_size == last_file_size:
        return False  # No changes
    
    rewards = evolution_data["rewards"].copy()
    agents = evolution_data["agents"].copy()
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Parse evaluations
        if "EVALUATION" in line:
            agent_id = None
            reward = None
            task_content = None
            
            # Look for agent details in the next few lines
            for j in range(i+1, min(i+10, len(lines))):
                if j >= len(lines):
                    break
                    
                if "Agent ID:" in lines[j]:
                    agent_id = lines[j].split("Agent ID:")[1].strip()
                elif "Reward:" in lines[j]:
                    try:
                        reward = float(lines[j].split("Reward:")[1].strip())
                        if reward not in rewards:  # Avoid duplicates
                            rewards.append(reward)
                    except ValueError:
                        pass
                elif "Task Chromosome" in lines[j]:
                    parts = lines[j].split(":")
                    if len(parts) > 1:
                        task_content = ":".join(parts[1:]).strip()
            
            if agent_id and reward is not None:
                # Check if this agent is already in our list
                agent_exists = False
                for agent in agents:
                    if agent["id"] == agent_id:
                        agent["reward"] = reward
                        agent["task_content"] = task_content
                        agent_exists = True
                        break
                
                if not agent_exists:
                    agents.append({
                        "id": agent_id,
                        "reward": reward,
                        "task_content": task_content,
                        "task_length": len(task_content) if task_content else 0
                    })
        
        # Parse population stats
        elif "POPULATION STATS" in line:
            for j in range(i+1, min(i+15, len(lines))):
                if j >= len(lines):
                    break
                    
                if "Population size:" in lines[j]:
                    try:
                        evolution_data["stats"]["population_size"] = int(lines[j].split("Population size:")[1].strip())
                    except ValueError:
                        pass
                elif "Total evaluations:" in lines[j]:
                    try:
                        evolution_data["stats"]["total_evaluations"] = int(lines[j].split("Total evaluations:")[1].strip())
                    except ValueError:
                        pass
                elif "Mean reward:" in lines[j]:
                    try:
                        evolution_data["stats"]["mean_reward"] = float(lines[j].split("Mean reward:")[1].strip())
                    except ValueError:
                        pass
                elif "Median reward:" in lines[j]:
                    try:
                        evolution_data["stats"]["median_reward"] = float(lines[j].split("Median reward:")[1].strip())
                    except ValueError:
                        pass
                elif "Std deviation:" in lines[j]:
                    try:
                        evolution_data["stats"]["std_dev"] = float(lines[j].split("Std deviation:")[1].strip())
                    except ValueError:
                        pass
                elif "Best reward:" in lines[j]:
                    try:
                        evolution_data["stats"]["best_reward"] = float(lines[j].split("Best reward:")[1].strip())
                    except ValueError:
                        pass
                elif "Worst reward:" in lines[j]:
                    try:
                        evolution_data["stats"]["worst_reward"] = float(lines[j].split("Worst reward:")[1].strip())
                    except ValueError:
                        pass
                elif "Recent window stats" in lines[j]:
                    # Parse window stats in the next few lines
                    for k in range(j+1, min(j+5, len(lines))):
                        if k >= len(lines):
                            break
                            
                        if "Mean:" in lines[k]:
                            try:
                                evolution_data["window_stats"]["mean"] = float(lines[k].split("Mean:")[1].strip())
                            except ValueError:
                                pass
                        elif "Median:" in lines[k]:
                            try:
                                evolution_data["window_stats"]["median"] = float(lines[k].split("Median:")[1].strip())
                            except ValueError:
                                pass
                        elif "Std deviation:" in lines[k]:
                            try:
                                evolution_data["window_stats"]["std_dev"] = float(lines[k].split("Std deviation:")[1].strip())
                            except ValueError:
                                pass
        
        i += 1
    
    # Update best agent
    if agents:
        best_agent = max(agents, key=lambda a: a["reward"])
        evolution_data["best_agent"] = best_agent
    
    # Update the data
    evolution_data["rewards"] = rewards
    evolution_data["agents"] = agents
    
    # Update file size
    last_file_size = current_size
    
    return True

def create_rewards_chart():
    """Create a chart of rewards over time"""
    rewards = evolution_data["rewards"]
    
    if not rewards:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot raw rewards
    ax.plot(rewards, 'o', alpha=0.3, color='lightgray', label='Individual rewards')
    
    # Calculate and plot moving average if we have enough data
    window_size = 10
    if len(rewards) >= window_size:
        moving_avg = []
        for i in range(len(rewards) - window_size + 1):
            moving_avg.append(np.mean(rewards[i:i+window_size]))
        
        # Plot moving average at the correct x positions
        ax.plot(range(window_size-1, len(rewards)), moving_avg, 
               linewidth=2, color='blue', label=f'Moving average (window={window_size})')
    
    # Add best reward line
    if rewards:
        best_reward = max(rewards)
        ax.axhline(y=best_reward, color='green', linestyle='--', 
                  label=f'Best reward: {best_reward:.2f}')
    
    # Add labels and title
    ax.set_xlabel('Evaluation Number')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Evolution Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_distribution_chart():
    """Create a histogram of reward distribution"""
    rewards = evolution_data["rewards"]
    
    if not rewards:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
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
    
    return fig

def create_top_agents_chart():
    """Create a bar chart of top agents by reward"""
    agents = evolution_data["agents"]
    
    if not agents:
        return None
    
    # Sort agents by reward and take top 10
    top_n = 10
    sorted_agents = sorted(agents, key=lambda x: x["reward"], reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Extract data for plotting
    ids = [agent["id"][:8] + '...' for agent in sorted_agents]
    values = [agent["reward"] for agent in sorted_agents]
    
    # Create bar chart
    bars = ax.bar(ids, values, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{height:.2f}', ha='center', va='bottom')
    
    # Add labels and title
    ax.set_xlabel('Agent ID')
    ax.set_ylabel('Reward')
    ax.set_title(f'Top {len(sorted_agents)} Agents by Reward')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    return fig

def update_data():
    """Update the data from the log file"""
    log_file = st.session_state.log_file
    if parse_log_file(log_file):
        st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        return True
    return False

def main():
    """Main Streamlit app"""
    # Sidebar
    st.sidebar.title("🧬 LLM Agent Evolution")
    st.sidebar.subheader("Dashboard Settings")
    
    # Log file selection
    log_files = glob.glob("*.log")
    default_log = "evolution.log" if "evolution.log" in log_files else (log_files[0] if log_files else "evolution.log")
    
    st.sidebar.selectbox(
        "Select Log File", 
        options=log_files,
        index=log_files.index(default_log) if default_log in log_files else 0,
        key="log_file"
    )
    
    # Auto-refresh toggle
    st.sidebar.checkbox("Auto-refresh", value=True, key="auto_refresh")
    
    # Refresh interval
    refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)", 
        min_value=1, 
        max_value=60, 
        value=5,
        key="refresh_interval"
    )
    
    # Manual refresh button
    if st.sidebar.button("Refresh Now"):
        update_data()
    
    # Last update time
    if "last_update" not in st.session_state:
        st.session_state.last_update = "Never"
    
    st.sidebar.text(f"Last update: {st.session_state.last_update}")
    
    # Main content
    st.title("LLM Agent Evolution Dashboard")
    
    # Statistics section
    st.header("Evolution Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Population Size", evolution_data["stats"]["population_size"])
        st.metric("Total Evaluations", evolution_data["stats"]["total_evaluations"])
    
    with col2:
        st.metric("Mean Reward", f"{evolution_data['stats']['mean_reward']:.2f}")
        st.metric("Median Reward", f"{evolution_data['stats']['median_reward']:.2f}")
    
    with col3:
        st.metric("Best Reward", f"{evolution_data['stats']['best_reward']:.2f}")
        st.metric("Std Deviation", f"{evolution_data['stats']['std_dev']:.2f}")
    
    # Recent window stats
    st.subheader("Recent Evaluations (Sliding Window)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Window Count", evolution_data["window_stats"]["count"])
    
    with col2:
        st.metric("Window Mean", f"{evolution_data['window_stats']['mean']:.2f}")
    
    with col3:
        st.metric("Window Std Dev", f"{evolution_data['window_stats']['std_dev']:.2f}")
    
    # Charts section
    st.header("Evolution Charts")
    
    tab1, tab2, tab3 = st.tabs(["Rewards Over Time", "Reward Distribution", "Top Agents"])
    
    with tab1:
        rewards_chart = create_rewards_chart()
        if rewards_chart:
            st.pyplot(rewards_chart)
        else:
            st.info("No reward data available yet.")
    
    with tab2:
        distribution_chart = create_distribution_chart()
        if distribution_chart:
            st.pyplot(distribution_chart)
        else:
            st.info("No reward data available yet.")
    
    with tab3:
        top_agents_chart = create_top_agents_chart()
        if top_agents_chart:
            st.pyplot(top_agents_chart)
        else:
            st.info("No agent data available yet.")
    
    # Best agent section
    st.header("Best Agent")
    
    if evolution_data["best_agent"]["id"]:
        st.subheader(f"ID: {evolution_data['best_agent']['id']}")
        st.metric("Reward", evolution_data["best_agent"]["reward"])
        
        st.text_area(
            "Task Chromosome", 
            evolution_data["best_agent"]["task_content"],
            height=150
        )
    else:
        st.info("No agent data available yet.")
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(0.1)  # Small delay to prevent UI freezing
        st.empty()  # Create a placeholder
        
        # Use a JavaScript hack to refresh the app
        st.markdown(
            f"""
            <script>
                var refresher = setInterval(function() {{
                    document.querySelector('button[kind=primaryFormSubmit]').click();
                }}, {st.session_state.refresh_interval * 1000});
            </script>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
