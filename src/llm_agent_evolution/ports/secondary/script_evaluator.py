"""
Script Evaluator Port - Interface for evaluating agent outputs using external scripts
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

class ScriptEvaluatorPort(ABC):
    """Secondary port for script-based evaluation"""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the evaluation cache
        
        Returns:
            Dictionary with cache statistics
        """
        pass
    
    @abstractmethod
    def clear_cache(self) -> None:
        """Clear the evaluation cache"""
        pass
