import os
import time
from typing import Dict, Any
from llm_agent_evolution.domain.model import Agent
from llm_agent_evolution.ports.secondary import LoggingPort

class FileLoggingAdapter(LoggingPort):
    """Adapter for logging to a file"""
    
    def __init__(self, log_file: str = "evolution.log"):
        """Initialize the logging adapter with the specified log file"""
        self.log_file = log_file
        self.evaluation_count = 0
        
        # Import tempfile here to avoid circular imports
        import tempfile
        self.temp_dir = tempfile.gettempdir()
    
    def initialize_log(self) -> None:
        """Initialize or clear the log file"""
        # Create an empty log file
        try:
            # Ensure directory exists with better error handling
            log_dir = os.path.dirname(os.path.abspath(self.log_file))
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    print(f"Could not create log directory {log_dir}: {e}")
                    # Fall back to temp directory
                    self.log_file = os.path.join(tempfile.gettempdir(), os.path.basename(self.log_file))
                    print(f"Using fallback log location: {self.log_file}")
            
            # Create or truncate the log file with explicit flush and sync
            try:
                with open(self.log_file, 'w') as f:
                    f.write(f"# LLM Agent Evolution Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"# This log contains detailed information about the evolution process\n")
                    f.write(f"# Format: timestamp | event_type | details\n\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                print(f"Could not write to log file {self.log_file}: {e}")
                # Fall back to temp directory
                self.log_file = os.path.join(tempfile.gettempdir(), f"evolution_{int(time.time())}.log")
                print(f"Using fallback log location: {self.log_file}")
                with open(self.log_file, 'w') as f:
                    f.write(f"# LLM Agent Evolution Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.flush()
                    os.fsync(f.fileno())
            
            # Double-check file was created and has content
            if not os.path.exists(self.log_file):
                print(f"Error: Log file {self.log_file} was not created")
                # Try creating in temp directory as fallback
                fallback_path = os.path.join(tempfile.gettempdir(), f"evolution_fallback_{int(time.time())}.log")
                with open(fallback_path, 'w') as f:
                    f.write(f"# LLM Agent Evolution Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.flush()
                self.log_file = fallback_path
                print(f"Created fallback log at: {fallback_path}")
                
            print(f"Log initialized at: {self.log_file}")
        except Exception as e:
            import traceback
            print(f"Error initializing log file {self.log_file}: {e}")
            print(traceback.format_exc())
            
            # Final fallback - use a guaranteed writable location
            try:
                fallback_path = os.path.join(tempfile.gettempdir(), f"evolution_emergency_{int(time.time())}.log")
                with open(fallback_path, 'w') as f:
                    f.write(f"# Emergency Fallback Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                self.log_file = fallback_path
                print(f"Created emergency fallback log at: {fallback_path}")
            except Exception:
                print("Could not create any log file. Logging will be disabled.")
    
    def log_evaluation(self, agent: Agent) -> None:
        """Log an agent evaluation"""
        self.evaluation_count += 1
        try:
            # Ensure the log file exists
            if not os.path.exists(self.log_file):
                self.initialize_log()
                
            with open(self.log_file, 'a') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EVALUATION #{self.evaluation_count}\n")
                f.write(f"Agent ID: {agent.id}\n")
                f.write(f"Reward: {agent.reward}\n")
                f.write(f"Task Chromosome ({len(agent.task_chromosome.content)} chars): {agent.task_chromosome.content[:50]}{'...' if len(agent.task_chromosome.content) > 50 else ''}\n")
                f.write(f"Mate Selection Chromosome ({len(agent.mate_selection_chromosome.content)} chars): {agent.mate_selection_chromosome.content[:50]}{'...' if len(agent.mate_selection_chromosome.content) > 50 else ''}\n")
                f.write(f"Mutation Chromosome ({len(agent.mutation_chromosome.content)} chars): {agent.mutation_chromosome.content[:50]}{'...' if len(agent.mutation_chromosome.content) > 50 else ''}\n")
                f.write("\n")
        except Exception as e:
            import traceback
            print(f"Error logging evaluation to {self.log_file}: {e}")
            print(traceback.format_exc())
    
    def log_population_stats(self, stats: Dict[str, Any]) -> None:
        """Log population statistics"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] POPULATION STATS\n")
                f.write(f"Population size: {stats.get('population_size', 0)}\n")
                f.write(f"Total evaluations: {stats.get('count', 0)}\n")
                
                if stats.get('mean') is not None:
                    f.write(f"Mean reward: {stats.get('mean'):.2f}\n")
                    f.write(f"Median reward: {stats.get('median'):.2f}\n")
                    f.write(f"Std deviation: {stats.get('std_dev'):.2f}\n")
                    f.write(f"Best reward: {stats.get('best'):.2f}\n")
                    f.write(f"Worst reward: {stats.get('worst'):.2f}\n")
                
                # Window stats
                window_stats = stats.get('window_stats', {})
                if window_stats and window_stats.get('count', 0) > 0:
                    f.write(f"Recent window stats (last {window_stats.get('window_size', 100)}):\n")
                    f.write(f"  Mean: {window_stats.get('mean'):.2f}\n")
                    f.write(f"  Median: {window_stats.get('median'):.2f}\n")
                    f.write(f"  Std deviation: {window_stats.get('std_dev'):.2f}\n")
                
                f.write("\n")
        except Exception as e:
            print(f"Warning: Could not log population stats to {self.log_file}: {e}")
    
    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log a general event"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {event_type.upper()}\n")
                for key, value in details.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
        except Exception as e:
            print(f"Warning: Could not log event to {self.log_file}: {e}")
