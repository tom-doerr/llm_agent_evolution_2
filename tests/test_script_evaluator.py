import pytest
import sys
import os
import tempfile
import time

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_agent_evolution.adapters.secondary.script_evaluator import ScriptEvaluatorAdapter

def create_test_script(content):
    """Create a temporary test script with the given content"""
    script_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    script_file.write(content)
    script_file.close()
    os.chmod(script_file.name, 0o755)  # Make executable
    return script_file.name

def test_script_evaluator_basic():
    """Test basic script evaluation"""
    # Create a simple test script
    script_content = """#!/usr/bin/env python3
import sys
text = sys.stdin.read()
print(len(text))  # Reward is the length of the text
"""
    script_path = create_test_script(script_content)
    
    try:
        # Create evaluator
        evaluator = ScriptEvaluatorAdapter()
        
        # Test evaluation
        reward = evaluator.evaluate("hello", script_path)
        assert reward == 5.0  # Length of "hello"
        
        # Test another input
        reward = evaluator.evaluate("hello world", script_path)
        assert reward == 11.0  # Length of "hello world"
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)

def test_script_evaluator_caching():
    """Test that caching works"""
    # Create a simple test script
    script_content = """#!/usr/bin/env python3
import sys
import time
text = sys.stdin.read()
# Add a delay to simulate computation
time.sleep(0.1)
print(len(text))
"""
    script_path = create_test_script(script_content)
    
    try:
        # Create evaluator
        evaluator = ScriptEvaluatorAdapter()
        
        # First evaluation (should be slow)
        start_time = time.time()
        reward1 = evaluator.evaluate("test", script_path)
        first_duration = time.time() - start_time
        
        # Second evaluation with same input (should be fast due to caching)
        start_time = time.time()
        reward2 = evaluator.evaluate("test", script_path)
        second_duration = time.time() - start_time
        
        # Check results
        assert reward1 == reward2 == 4.0  # Length of "test"
        
        # Check that second evaluation was faster, with some tolerance for system variations
        assert second_duration < first_duration * 0.9 or second_duration < 0.01, \
            f"Second evaluation ({second_duration:.4f}s) should be faster than first ({first_duration:.4f}s)"
        
        # Check cache stats
        stats = evaluator.get_cache_stats()
        assert stats["hits"] >= 1, "Cache should have at least one hit"
        assert stats["misses"] >= 1, "Cache should have at least one miss"
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)

def test_script_evaluator_batch():
    """Test batch evaluation"""
    # Create a simple test script
    script_content = """#!/usr/bin/env python3
import sys
text = sys.stdin.read()
print(len(text))
"""
    script_path = create_test_script(script_content)
    
    try:
        # Create evaluator
        evaluator = ScriptEvaluatorAdapter()
        
        # Test batch evaluation
        outputs = ["a", "bb", "ccc", "dddd"]
        rewards = evaluator.evaluate_batch(outputs, script_path, parallel=True)
        
        # Check results
        assert rewards == [1.0, 2.0, 3.0, 4.0]
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)

def test_script_evaluator_error_handling():
    """Test error handling"""
    # Create a script that raises an error
    script_content = """#!/usr/bin/env python3
import sys
raise ValueError("Test error")
"""
    script_path = create_test_script(script_content)
    
    try:
        # Create evaluator
        evaluator = ScriptEvaluatorAdapter()
        
        # Test evaluation with error
        with pytest.raises(RuntimeError) as excinfo:
            evaluator.evaluate("test", script_path)
        
        # Verify the error message contains the original error
        assert "Test error" in str(excinfo.value) or "ValueError" in str(excinfo.value)
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)
