"""
Script Evaluator Adapter - Implementation of script-based evaluation
"""
import os
import sys
import subprocess
import threading
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
import tempfile
from concurrent.futures import ThreadPoolExecutor

from llm_agent_evolution.ports.secondary.script_evaluator import ScriptEvaluatorPort

class ScriptEvaluatorAdapter(ScriptEvaluatorPort):
    """Adapter for evaluating agent outputs using external scripts"""
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        """
        Initialize the script evaluator adapter
        
        Args:
            cache_size: Maximum number of cached results
            cache_ttl: Time-to-live for cache entries in seconds
        """
        self.cache = {}  # {hash: (reward, timestamp)}
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.Lock()
    
    def _hash_input(self, output: str, script_path: str) -> str:
        """Generate a hash for the output and script path"""
        script_hash = hashlib.md5(open(script_path, 'rb').read()).hexdigest()
        output_hash = hashlib.md5(output.encode('utf-8')).hexdigest()
        return f"{script_hash}_{output_hash}"
    
    def _check_cache(self, output: str, script_path: str) -> Tuple[bool, Optional[float]]:
        """Check if result is in cache and still valid"""
        with self.lock:
            input_hash = self._hash_input(output, script_path)
            if input_hash in self.cache:
                reward, timestamp = self.cache[input_hash]
                if time.time() - timestamp <= self.cache_ttl:
                    self.cache_hits += 1
                    return True, reward
            self.cache_misses += 1
            return False, None
    
    def _update_cache(self, output: str, script_path: str, reward: float) -> None:
        """Update the cache with a new result"""
        with self.lock:
            input_hash = self._hash_input(output, script_path)
            self.cache[input_hash] = (reward, time.time())
            
            # Prune cache if it exceeds the size limit
            if len(self.cache) > self.cache_size:
                # Remove oldest entries
                sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_items[:len(self.cache) - self.cache_size]:
                    del self.cache[key]
    
    def evaluate(self, output: str, script_path: str, timeout: int = 30) -> float:
        """
        Evaluate the output using the specified script
        
        Args:
            output: The text output to evaluate
            script_path: Path to the evaluation script
            timeout: Maximum execution time in seconds
            
        Returns:
            Numerical reward value
        """
        # Check cache first
        in_cache, cached_reward = self._check_cache(output, script_path)
        if in_cache:
            return cached_reward
        
        # Ensure script exists and is executable
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Evaluation script not found: {script_path}")
        
        if not os.access(script_path, os.X_OK) and script_path.endswith('.py'):
            # For Python scripts, we can run them with the Python interpreter
            cmd = [sys.executable, script_path]
        else:
            # Make sure the script is executable
            os.chmod(script_path, 0o755)
            cmd = [script_path]
        
        # Write output to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(output)
        
        try:
            # Run the script with the output as stdin
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send the output to the script
            stdout, stderr = process.communicate(input=output, timeout=timeout)
            
            # Check for errors
            if process.returncode != 0:
                raise RuntimeError(f"Evaluation script failed with error: {stderr}")
            
            # Parse the reward from the last line of output
            lines = stdout.strip().split('\n')
            if not lines:
                raise ValueError("Evaluation script produced no output")
            
            try:
                reward = float(lines[-1].strip())
            except ValueError:
                raise ValueError(f"Evaluation script did not return a valid numerical reward: {lines[-1]}")
            
            # Update cache
            self._update_cache(output, script_path, reward)
            
            return reward
            
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Evaluation script timed out after {timeout} seconds")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    def evaluate_batch(self, outputs: List[str], script_path: str, 
                      timeout: int = 30, parallel: bool = True) -> List[float]:
        """
        Evaluate multiple outputs using the specified script
        
        Args:
            outputs: List of text outputs to evaluate
            script_path: Path to the evaluation script
            timeout: Maximum execution time in seconds
            parallel: Whether to run evaluations in parallel
            
        Returns:
            List of numerical reward values
        """
        if not parallel:
            # Sequential evaluation
            return [self.evaluate(output, script_path, timeout) for output in outputs]
        
        # Parallel evaluation
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.evaluate, output, script_path, timeout)
                for output in outputs
            ]
            
            # Collect results, handling exceptions
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    # Log the error and assign a zero reward
                    print(f"Evaluation error: {e}")
                    results.append(0.0)
            
            return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the evaluation cache
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.cache_size,
                "ttl": self.cache_ttl,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            }
    
    def clear_cache(self) -> None:
        """Clear the evaluation cache"""
        with self.lock:
            self.cache.clear()
            # Keep the hit/miss statistics
