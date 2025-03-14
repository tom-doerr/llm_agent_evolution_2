# Universal Optimizer

A flexible, command-based optimization framework for evolving text outputs against any measurable goal.

## Overview

The Universal Optimizer allows you to optimize any text-based output using evolutionary algorithms. It works by:

1. Generating a population of text outputs
2. Evaluating each output using your custom command or script
3. Evolving the population through selection, mating, and mutation
4. Repeating until optimal solutions are found

The key innovation is the command-based evaluation interface, which allows you to define any optimization goal by providing a command that returns a numerical reward.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm_agent_evolution.git
cd llm_agent_evolution

# Install dependencies
pip install -e .
```

## Quick Start

You can use the Universal Optimizer in two ways:

### Option 1: Direct Command

Provide an evaluation command directly:

```bash
./universal_optimize.py "python -c 'import sys; text=sys.stdin.read(); print(len(text))'" --population-size 50 --max-evaluations 1000
```

This command evaluates text by its length.

### Option 2: Evaluation Script

1. Create an evaluation script that takes text input via stdin and outputs a numerical reward as its last line:

```python
#!/usr/bin/env python3
import sys

# Read input from stdin
text = sys.stdin.read()

# Calculate reward (higher is better)
reward = len(text)  # Simple example: reward by length

# Print reward as the last line
print(reward)
```

2. Make the script executable:

```bash
chmod +x my_eval_script.py
```

3. Run the optimizer:

```bash
./universal_optimize.py --eval-script my_eval_script.py --population-size 50 --max-evaluations 1000
```

## Command Line Options

```
EVAL_COMMAND            Evaluation command (receives agent output via stdin, returns score as last line)
--eval-script SCRIPT     Path to the evaluation script (alternative to EVAL_COMMAND)
--population-size N      Initial population size (default: 50)
--parallel-agents N      Number of agents to evaluate in parallel (default: 8)
--max-evaluations N      Maximum number of evaluations to run (default: unlimited)
--use-mock-llm           Use mock LLM adapter for testing
--model MODEL            LLM model to use (default: openrouter/google/gemini-2.0-flash-001)
--log-file FILE          Log file path (default: universal_optimize.log)
--seed SEED              Random seed for reproducibility
--script-timeout SEC     Maximum execution time for the evaluation script (default: 30)
--initial-content TEXT   Initial content for the chromosomes
--initial-file FILE      File containing initial content for the chromosomes
--output-file FILE       File to write the best result to
--output-format FORMAT   Output format: text or json (default: text)
--max-chars N            Maximum number of characters for chromosomes (default: 1000)
--verbose                Enable verbose output (limited to first 5 agents)
```

## Evaluation Scripts

The evaluation script is the heart of the Universal Optimizer. It should:

1. Read input from stdin
2. Process the input in any way you want
3. Output a numerical reward as the last line of stdout

The reward should be higher for better solutions. The optimizer will try to maximize this value.

### Example Evaluation Scripts

#### Count 'a's in Text

```python
#!/usr/bin/env python3
import sys

text = sys.stdin.read()
reward = text.count('a')
print(reward)
```

#### Test Code Against Test Cases

```python
#!/usr/bin/env python3
import sys
import ast

# Read code from stdin
code = sys.stdin.read()

# Define test cases
test_cases = [
    (5, 120),    # factorial(5) should be 120
    (0, 1),      # factorial(0) should be 1
    (1, 1),      # factorial(1) should be 1
    (10, 3628800) # factorial(10) should be 3628800
]

# Try to execute the code
try:
    # Check syntax
    ast.parse(code)
    
    # Create namespace and execute code
    namespace = {}
    exec(code, namespace)
    
    # Check if factorial function exists
    if 'factorial' not in namespace:
        print("Function 'factorial' not found")
        print(0)  # Reward
        sys.exit(0)
    
    # Run test cases
    passing = 0
    for input_val, expected in test_cases:
        try:
            result = namespace['factorial'](input_val)
            if result == expected:
                passing += 1
        except Exception:
            pass
    
    # Calculate reward
    reward = passing / len(test_cases) * 10
    
    # Print results and reward
    print(f"Passing tests: {passing}/{len(test_cases)}")
    print(reward)
    
except Exception as e:
    print(f"Error: {e}")
    print(0)  # Reward
```

#### Optimize Text Readability

```python
#!/usr/bin/env python3
import sys
import textstat

text = sys.stdin.read()

# Calculate readability metrics
flesch_reading_ease = textstat.flesch_reading_ease(text)
flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)

# Normalize and combine metrics
# Higher Flesch Reading Ease is better (easier to read)
# Lower Flesch-Kincaid Grade is better (lower grade level required)
normalized_grade = max(0, 20 - flesch_kincaid_grade) / 20
normalized_ease = flesch_reading_ease / 100

# Calculate combined reward
reward = (normalized_grade + normalized_ease) * 5

print(f"Flesch Reading Ease: {flesch_reading_ease}")
print(f"Flesch-Kincaid Grade: {flesch_kincaid_grade}")
print(reward)
```

## Advanced Usage

### Using Initial Content

You can provide initial content to seed the optimization:

```bash
./universal_optimize.py --eval-script my_eval_script.py --initial-content "Starting text"
```

Or from a file:

```bash
./universal_optimize.py --eval-script my_eval_script.py --initial-file my_starting_point.txt
```

### Saving Results

Save the best result to a file:

```bash
./universal_optimize.py --eval-script my_eval_script.py --output-file best_result.txt
```

Save detailed results in JSON format:

```bash
./universal_optimize.py --eval-script my_eval_script.py --output-file results.json --output-format json
```

### Using with DSPy

You can use the Universal Optimizer to optimize DSPy prompts:

```python
#!/usr/bin/env python3
import sys
import dspy

# Read prompt from stdin
prompt = sys.stdin.read()

# Create DSPy program with the prompt
lm = dspy.LM('openrouter/google/gemini-2.0-flash-001')
program = dspy.Predict("Question -> Answer", prompt)

# Define test cases
test_cases = [
    {"Question": "What is the capital of France?", "Answer": "Paris"},
    {"Question": "Who wrote Romeo and Juliet?", "Answer": "William Shakespeare"},
    # Add more test cases...
]

# Evaluate the prompt
correct = 0
for test_case in test_cases:
    try:
        prediction = program(dspy.Example(Question=test_case["Question"]))
        if test_case["Answer"].lower() in prediction.Answer.lower():
            correct += 1
    except Exception:
        pass

# Calculate reward
reward = correct / len(test_cases) * 10

# Print results and reward
print(f"Correct answers: {correct}/{len(test_cases)}")
print(reward)
```

## How It Works

The Universal Optimizer uses an evolutionary algorithm with:

1. **Population Initialization**: Creates a population of text outputs
2. **Evaluation**: Uses your script to evaluate each output
3. **Selection**: Selects parents based on their rewards
4. **Mating**: Combines parent outputs to create offspring
5. **Mutation**: Introduces variations using an LLM
6. **Replacement**: Adds new outputs to the population, removing worst ones if needed

The process continues until the maximum number of evaluations is reached or you stop it manually.

## Tips for Effective Optimization

1. **Design good reward functions**: The reward should guide the optimizer toward your goal
2. **Start with small populations**: Begin with 20-50 agents and increase if needed
3. **Use initial content**: Provide a starting point to speed up optimization
4. **Monitor progress**: Watch the mean and best rewards to see if optimization is working
5. **Adjust timeout**: Set an appropriate script timeout based on your evaluation complexity
6. **Use the mock LLM**: For testing your setup before using real API calls

## License

MIT
