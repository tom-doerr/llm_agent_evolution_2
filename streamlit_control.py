#!/usr/bin/env python3
"""
Streamlit app for controlling LLM Agent Evolution
"""
import os
import sys
import time
import glob
import subprocess
import threading
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from llm_agent_evolution.domain.model import Agent, Chromosome
from llm_agent_evolution.adapters.secondary.mock_llm import MockLLMAdapter
from llm_agent_evolution.application import create_application

# Set page configuration
st.set_page_config(
    page_title="LLM Agent Evolution Control Center",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables
evolution_process = None
stop_event = threading.Event()

def start_evolution_process(args):
    """Start the evolution process with the given arguments"""
    global evolution_process
    
    # Build the command
    cmd = ["python", "-m", "llm_agent_evolution"]
    
    # Add arguments
    if args.get("population_size"):
        cmd.extend(["--population-size", str(args["population_size"])])
    if args.get("parallel_agents"):
        cmd.extend(["--parallel-agents", str(args["parallel_agents"])])
    if args.get("max_evaluations"):
        cmd.extend(["--max-evaluations", str(args["max_evaluations"])])
    if args.get("model"):
        cmd.extend(["--model", args["model"]])
    if args.get("log_file"):
        cmd.extend(["--log-file", args["log_file"]])
    if args.get("use_mock"):
        cmd.append("--use-mock")
    if args.get("seed") is not None:
        cmd.extend(["--seed", str(args["seed"])])
    if args.get("no_visualization"):
        cmd.append("--no-visualization")
    
    # Set environment variables
    env = os.environ.copy()
    
    # Start the process
    try:
        evolution_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        st.session_state.process_running = True
        st.session_state.process_output = []
        
        # Start a thread to read output
        threading.Thread(
            target=read_process_output,
            args=(evolution_process, stop_event),
            daemon=True
        ).start()
        
        return True
    except Exception as e:
        st.error(f"Failed to start evolution process: {e}")
        return False

def read_process_output(process, stop_event):
    """Read output from the process and store it"""
    while process.poll() is None and not stop_event.is_set():
        line = process.stdout.readline()
        if line:
            with st.session_state.lock:
                st.session_state.process_output.append(line.strip())
        time.sleep(0.1)
    
    # Read any remaining output
    remaining_output, remaining_error = process.communicate()
    if remaining_output:
        with st.session_state.lock:
            for line in remaining_output.splitlines():
                st.session_state.process_output.append(line.strip())
    
    # Process has ended
    with st.session_state.lock:
        st.session_state.process_running = False

def stop_evolution_process():
    """Stop the evolution process"""
    global evolution_process, stop_event
    
    if evolution_process and evolution_process.poll() is None:
        stop_event.set()
        evolution_process.terminate()
        evolution_process.wait(timeout=5)
        if evolution_process.poll() is None:
            evolution_process.kill()
        
        st.session_state.process_running = False
        return True
    
    return False

def create_custom_task():
    """Create a custom task for evolution"""
    st.subheader("Custom Task Creator")
    
    task_type = st.selectbox(
        "Task Type",
        ["Text Generation", "Classification", "Optimization", "Custom"]
    )
    
    if task_type == "Text Generation":
        st.write("Define a text generation task:")
        target_text = st.text_area("Target Text Pattern", "")
        max_length = st.number_input("Maximum Length", min_value=1, value=100)
        
        if st.button("Create Text Generation Task"):
            # Create a custom LLM adapter for this task
            task_code = f"""
# Custom Text Generation Task
# Target: {target_text}
# Max Length: {max_length}

class CustomLLMAdapter(LLMPort):
    def evaluate_task_output(self, output: str) -> float:
        # Count matches to target pattern
        import re
        pattern = r"{target_text}"
        matches = len(re.findall(pattern, output))
        
        # Penalty for exceeding max length
        length_penalty = max(0, len(output) - {max_length})
        
        # Calculate reward
        reward = matches - (length_penalty * 0.1)
        return max(0, reward)
"""
            st.code(task_code, language="python")
            st.success("Task created! Copy this code to create a custom adapter.")
    
    elif task_type == "Classification":
        st.write("Define a classification task:")
        categories = st.text_area("Categories (one per line)", "")
        
        if st.button("Create Classification Task"):
            categories_list = [c.strip() for c in categories.split("\n") if c.strip()]
            task_code = f"""
# Custom Classification Task
# Categories: {categories_list}

class CustomLLMAdapter(LLMPort):
    def evaluate_task_output(self, output: str) -> float:
        categories = {categories_list}
        
        # Check if output matches any category
        output_lower = output.lower()
        for category in categories:
            if category.lower() in output_lower:
                return 1.0
        
        return 0.0
"""
            st.code(task_code, language="python")
            st.success("Task created! Copy this code to create a custom adapter.")
    
    elif task_type == "Optimization":
        st.write("Define an optimization task:")
        target_value = st.number_input("Target Value", value=42)
        
        if st.button("Create Optimization Task"):
            task_code = f"""
# Custom Optimization Task
# Target Value: {target_value}

class CustomLLMAdapter(LLMPort):
    def evaluate_task_output(self, output: str) -> float:
        try:
            # Try to extract a number from the output
            import re
            numbers = re.findall(r'\\d+\\.?\\d*', output)
            if not numbers:
                return 0.0
            
            # Use the first number found
            value = float(numbers[0])
            
            # Calculate distance from target
            distance = abs(value - {target_value})
            
            # Convert to reward (closer is better)
            reward = 1.0 / (1.0 + distance)
            return reward
        except:
            return 0.0
"""
            st.code(task_code, language="python")
            st.success("Task created! Copy this code to create a custom adapter.")
    
    else:  # Custom
        st.write("Define a custom task:")
        reward_function = st.text_area(
            "Reward Function (Python code)",
            """def evaluate_task_output(output: str) -> float:
    # Your custom evaluation logic here
    # Return a float representing the reward
    return 0.0"""
        )
        
        if st.button("Create Custom Task"):
            task_code = f"""
# Custom Task

class CustomLLMAdapter(LLMPort):
{reward_function.replace('def evaluate_task_output', '    def evaluate_task_output')}
"""
            st.code(task_code, language="python")
            st.success("Task created! Copy this code to create a custom adapter.")

def main():
    """Main Streamlit app"""
    # Initialize session state
    if "process_running" not in st.session_state:
        st.session_state.process_running = False
    if "process_output" not in st.session_state:
        st.session_state.process_output = []
    if "lock" not in st.session_state:
        st.session_state.lock = threading.Lock()
    
    # Sidebar
    st.sidebar.title("🧬 LLM Agent Evolution")
    st.sidebar.subheader("Control Center")
    
    # Evolution parameters
    st.sidebar.subheader("Evolution Parameters")
    
    population_size = st.sidebar.number_input(
        "Population Size",
        min_value=10,
        max_value=1000,
        value=50,
        step=10
    )
    
    parallel_agents = st.sidebar.number_input(
        "Parallel Agents",
        min_value=1,
        max_value=32,
        value=8,
        step=1
    )
    
    max_evaluations = st.sidebar.number_input(
        "Max Evaluations",
        min_value=0,
        value=1000,
        step=100,
        help="0 for unlimited"
    )
    
    use_mock = st.sidebar.checkbox("Use Mock LLM", value=True)
    
    model = st.sidebar.text_input(
        "LLM Model",
        value="openrouter/google/gemini-2.0-flash-001",
        disabled=use_mock
    )
    
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        value=42,
        step=1,
        help="For reproducibility"
    )
    
    log_file = st.sidebar.text_input(
        "Log File",
        value="evolution.log"
    )
    
    # Control buttons
    st.sidebar.subheader("Control")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Start Evolution", disabled=st.session_state.process_running):
            args = {
                "population_size": population_size,
                "parallel_agents": parallel_agents,
                "max_evaluations": max_evaluations if max_evaluations > 0 else None,
                "model": model,
                "log_file": log_file,
                "use_mock": use_mock,
                "seed": seed,
                "no_visualization": False
            }
            start_evolution_process(args)
    
    with col2:
        if st.button("Stop Evolution", disabled=not st.session_state.process_running):
            stop_evolution_process()
    
    # Quick test button
    if st.sidebar.button("Run Quick Test", disabled=st.session_state.process_running):
        args = {
            "population_size": 20,
            "parallel_agents": 4,
            "max_evaluations": 100,
            "use_mock": True,
            "seed": seed,
            "log_file": "quick_test.log"
        }
        start_evolution_process(args)
    
    # Main content
    st.title("LLM Agent Evolution Control Center")
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Evolution Control", "Custom Tasks", "Documentation"])
    
    with tab1:
        # Status
        st.subheader("Evolution Status")
        
        status = "Running" if st.session_state.process_running else "Stopped"
        st.info(f"Status: {status}")
        
        # Process output
        st.subheader("Process Output")
        
        output_container = st.container()
        with output_container:
            # Display the last 20 lines of output
            for line in st.session_state.process_output[-20:]:
                st.text(line)
        
        # Auto-refresh for process output
        if st.session_state.process_running:
            st.empty()
            time.sleep(1)
            st.experimental_rerun()
        
        # Link to dashboard
        st.subheader("Monitoring")
        st.write("Open the dashboard to monitor evolution in real-time:")
        if st.button("Open Dashboard"):
            # Use JavaScript to open the dashboard in a new tab
            js = f"""
            <script>
                window.open("http://localhost:8765", "_blank");
            </script>
            """
            st.components.v1.html(js, height=0)
    
    with tab2:
        create_custom_task()
    
    with tab3:
        st.subheader("Documentation")
        
        st.markdown("""
        ## LLM Agent Evolution Framework
        
        This framework allows you to evolve LLM-based agents through evolutionary algorithms.
        
        ### Key Concepts
        
        - **Agent**: An entity with three chromosomes (task, mate selection, mutation)
        - **Chromosome**: A string of text representing a specific function
        - **Evolution**: Continuous process of selection, mating, and mutation
        
        ### Suggested Evolution Tasks
        
        1. **Text Pattern Generation**: Evolve agents to generate specific text patterns
        2. **Code Generation**: Evolve agents to write simple code snippets
        3. **Mathematical Problem Solving**: Evolve agents to solve math problems
        4. **Creative Writing**: Evolve agents to write poetry or stories
        5. **Logical Reasoning**: Evolve agents to solve logical puzzles
        
        ### Custom Task Creation
        
        To create a custom task:
        
        1. Define a reward function that evaluates agent outputs
        2. Create a custom LLM adapter that implements this function
        3. Wire the adapter into the application
        
        See the "Custom Tasks" tab for examples.
        """)

if __name__ == "__main__":
    main()
