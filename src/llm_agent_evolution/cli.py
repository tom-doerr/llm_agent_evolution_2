import sys
import os
import argparse
import tempfile
from typing import List, Optional

from llm_agent_evolution.evolution import run_optimizer, load_agent, evaluate_agent_with_command

def main(args: Optional[List[str]] = None) -> int:
    try:
        # Parse arguments
        parser, parsed_args = _create_parser_and_parse_args(args)
        
        # Process context from various sources
        context = _get_context(parsed_args)
        if context:
            os.environ['AGENT_CONTEXT'] = context
        
        # Handle different execution modes
        if parsed_args.load and parsed_args.eval_command:
            return _run_loaded_agent(parsed_args, context)
        elif parsed_args.eval_command:
            return _run_optimizer(parsed_args, context)
        elif parsed_args.quick_test:
            return _run_quick_test(parsed_args)
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
        return 1

def _create_parser_and_parse_args(args: Optional[List[str]]) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    # Check for positional eval_command
    eval_command = None
    if args is None and len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        eval_command = sys.argv[1]
        sys.argv.pop(1)
    
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="LLM Agent Evolution - Evolve LLM agents through natural selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core parameters
    parser.add_argument("--population-size", "-p", type=int, default=100)
    parser.add_argument("--parallel-agents", "-j", type=int, default=10)
    parser.add_argument("--max-evaluations", "-n", type=int, default=None)
    parser.add_argument("--model", "-m", type=str, default="openrouter/google/gemini-2.0-flash-001")
    parser.add_argument("--use-mock", "--mock", action="store_true")
    parser.add_argument("--eval-command", "-e", type=str, default=None)
    parser.add_argument("--load", "-l", type=str, default=None)
    parser.add_argument("--save", "-o", type=str, default=None)
    parser.add_argument("--context", "-c", type=str, default=None)
    parser.add_argument("--context-file", "-cf", type=str, default=None)
    parser.add_argument("--initial-content", "-i", type=str, default="")
    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument("--quick-test", "-q", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # If we found a positional eval_command, add it to the parsed args
    if eval_command:
        parsed_args.eval_command = eval_command
    
    # Handle quick test mode
    if parsed_args.quick_test:
        parsed_args.use_mock = True
        if not parsed_args.max_evaluations:
            parsed_args.max_evaluations = 100
        parsed_args.population_size = 20
        parsed_args.parallel_agents = 4
        if parsed_args.seed is None:
            parsed_args.seed = 42
    
    return parser, parsed_args

def _get_context(args: argparse.Namespace) -> Optional[str]:
    # Get context from file if specified
    context = args.context
    if args.context_file:
        try:
            with open(args.context_file, 'r') as f:
                context = f.read()
        except Exception as e:
            print(f"Error reading context file: {e}")
            return None
    
    # Check if we should read from stdin
    if context is None and not sys.stdin.isatty():
        try:
            context = sys.stdin.read()
            print(f"Read context from stdin: {context[:50]}{'...' if len(context) > 50 else ''}")
        except Exception as e:
            print(f"Error reading from stdin: {e}")
    
    return context

def _run_loaded_agent(args: argparse.Namespace, context: Optional[str]) -> int:
    agent = load_agent(args.load)
    if not agent:
        print(f"Error: Could not load agent from {args.load}")
        return 1
        
    print(f"Loaded agent with ID: {agent.id}")
    print(f"Agent content: {agent.task_chromosome.content[:50]}...")
    
    reward = evaluate_agent_with_command(agent, args.eval_command, context)
    print(f"\nAgent output: {agent.task_chromosome.content}")
    return 0

def _run_optimizer(args: argparse.Namespace, context: Optional[str]) -> int:
    result = run_optimizer(
        eval_command=args.eval_command,
        population_size=args.population_size,
        parallel_agents=args.parallel_agents,
        max_evaluations=args.max_evaluations,
        use_mock_llm=args.use_mock,
        model_name=args.model,
        initial_content=args.initial_content,
        verbose=args.verbose,
        random_seed=args.seed
    )
    
    # Save the best agent if requested
    if args.save and result["best_agent"]["content"]:
        from llm_agent_evolution.evolution import save_agent
        from llm_agent_evolution.domain.model import Agent, Chromosome
        
        best_agent = Agent(
            task_chromosome=Chromosome(content=result["best_agent"]["content"], type="task"),
            mate_selection_chromosome=Chromosome(content="", type="mate_selection"),
            mutation_chromosome=Chromosome(content="", type="mutation"),
            id=result["best_agent"]["id"],
            reward=result["best_agent"]["reward"]
        )
        
        if save_agent(best_agent, args.save):
            print(f"\nBest agent saved to: {args.save}")
    
    return 0

def _run_quick_test(args: argparse.Namespace) -> int:
    from llm_agent_evolution.quick_test import main as run_quick_test
    return run_quick_test(seed=args.seed)
def _create_main_parser():
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        description="LLM Agent Evolution - A framework for evolving LLM-based agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add the main arguments to the top-level parser
    _add_common_arguments(parser)
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add subcommand parsers
    _add_evolve_subparser(subparsers)
    _add_optimize_subparser(subparsers)
    _add_standalone_subparser(subparsers)
    _add_demo_subparser(subparsers)
    _add_inference_subparser(subparsers)
    
    return parser

def _add_common_arguments(parser):
    """Add common arguments to a parser"""
    # Helper function to add argument only if it doesn't exist
    def add_arg_if_not_exists(name, *args, **kwargs):
        if not any(action.dest == name for action in parser._actions):
            parser.add_argument(*args, **kwargs)
    
    add_arg_if_not_exists('population_size',
        "--population-size", "-p",
        type=int, 
        default=100,
        help="Initial population size"
    )
    
    add_arg_if_not_exists('parallel_agents',
        "--parallel-agents", "-j",
        type=int, 
        default=10,
        help="Number of agents to evaluate in parallel"
    )
    
    add_arg_if_not_exists('max_evaluations',
        "--max-evaluations", "-n",
        type=int, 
        default=None,
        help="Maximum number of evaluations to run"
    )
    
    add_arg_if_not_exists('use_mock',
        "--use-mock", "--mock",
        action="store_true",
        help="Use mock LLM adapter for testing"
    )
    
    add_arg_if_not_exists('model',
        "--model", "-m",
        type=str, 
        default="openrouter/google/gemini-2.0-flash-001",
        help="LLM model to use"
    )
    
    add_arg_if_not_exists('log_file',
        "--log-file",
        type=str, 
        default="evolution.log",
        help="Log file path"
    )
    
    add_arg_if_not_exists('seed',
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    add_arg_if_not_exists('eval_command',
        "--eval-command", "-e",
        type=str,
        default=None,
        help="Command to run for evaluation"
    )
    
    add_arg_if_not_exists('quick_test',
        "--quick-test", "-q",
        action="store_true",
        help="Run a quick test with mock LLM"
    )
    
    add_arg_if_not_exists('save',
        "--save", "-o",
        type=str,
        default=None,
        help="File to save the best result to"
    )
    
    add_arg_if_not_exists('load',
        "--load", "-l",
        type=str,
        default=None,
        help="Load a previously saved agent from file"
    )
    
    add_arg_if_not_exists('context',
        "--context", "-c",
        type=str,
        default=None,
        help="Context to pass to the agent (available as AGENT_CONTEXT environment variable)"
    )
    
    add_arg_if_not_exists('context_file',
        "--context-file", "-cf",
        type=str,
        default=None,
        help="File containing context to pass to the agent"
    )
    
    add_arg_if_not_exists('initial_content',
        "--initial-content", "-i",
        type=str,
        default="",
        help="Initial content for the chromosomes"
    )
    
    add_arg_if_not_exists('verbose',
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

def _add_evolve_subparser(subparsers):
    """Add the evolve subparser"""
    evolve_parser = subparsers.add_parser(
        "evolve", 
        help="Run the main evolution process"
    )
    
    # Add arguments for the evolve command
    _add_common_arguments(evolve_parser)
    
def _add_optimize_subparser(subparsers):
    """Add the optimize subparser"""
    optimize_parser = subparsers.add_parser(
        "optimize", 
        help="Run the universal optimizer"
    )
    
    # Add arguments for the optimize command
    optimize_parser.add_argument(
        "eval_command",
        nargs="?",
        help="Evaluation command (receives agent output via stdin, returns score as last line)"
    )
    
    optimize_parser.add_argument(
        "--eval-script", 
        help="Path to the evaluation script (alternative to eval_command)"
    )
    
    # Add common arguments
    _add_common_arguments(optimize_parser)
    
    # Add optimize-specific arguments
    optimize_parser.add_argument(
        "--script-timeout", "-t",
        type=int,
        default=30,
        help="Maximum execution time for the evaluation script in seconds"
    )
    
    optimize_parser.add_argument(
        "--optimize-initial-content", "-I",
        type=str,
        default="",
        help="Initial content for the chromosomes"
    )
    
    optimize_parser.add_argument(
        "--initial-file", "-f",
        type=str,
        default=None,
        help="File containing initial content for the chromosomes"
    )
    
    optimize_parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "toml"],
        default="text",
        help="Output format (text or TOML)"
    )
    
    optimize_parser.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="Maximum number of characters for chromosomes"
    )
    
    # Verbose flag is added in _add_common_arguments
    
def _add_standalone_subparser(subparsers):
    """Add the standalone subparser"""
    standalone_parser = subparsers.add_parser(
        "standalone", 
        help="Run the simplified standalone optimizer (no LLM API calls)"
    )
    
    standalone_parser.add_argument(
        "eval_command",
        help="Evaluation command (receives agent output via stdin, returns score as last line)"
    )
    
    standalone_parser.add_argument(
        "--population-size", "-p",
        type=int, 
        default=50,
        help="Initial population size"
    )
    
    standalone_parser.add_argument(
        "--parallel-agents", "-j",
        type=int, 
        default=8,
        help="Number of agents to evaluate in parallel"
    )
    
    standalone_parser.add_argument(
        "--max-evaluations", "-n",
        type=int, 
        default=1000,
        help="Maximum number of evaluations to run"
    )
    
    standalone_parser.add_argument(
        "--standalone-initial-content", "-S",
        type=str,
        default="",
        help="Initial content for the chromosomes"
    )
    
    standalone_parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    # Verbose flag is added in _add_common_arguments
    
    standalone_parser.add_argument(
        "--save", "-o",
        type=str,
        default=None,
        help="File to save the best result to"
    )

def _add_demo_subparser(subparsers):
    """Add the demo subparser"""
    demo_parser = subparsers.add_parser(
        "demo", 
        help="Run the evolution demo with detailed step-by-step output"
    )
    
    demo_parser.add_argument(
        "--use-mock", "--mock",
        action="store_true",
        help="Use mock LLM instead of real LLM"
    )
    
    demo_parser.add_argument(
        "--demo-initial-content", "-D",
        type=str,
        default="a",
        help="Initial content for task chromosome"
    )

def _add_inference_subparser(subparsers):
    """Add the inference subparser"""
    inference_parser = subparsers.add_parser(
        "inference", 
        help="Run inference with a loaded agent (no evolution)"
    )
    
    inference_parser.add_argument(
        "--load", "-l",
        type=str,
        required=True,
        help="Load a previously saved agent from file"
    )
    
    inference_parser.add_argument(
        "--eval-command", "-e",
        type=str,
        required=True,
        help="Command to run for evaluation"
    )
    
    inference_parser.add_argument(
        "--context", "-c",
        type=str,
        default=None,
        help="Context to pass to the agent"
    )
    
    inference_parser.add_argument(
        "--context-file", "-cf",
        type=str,
        default=None,
        help="File containing context to pass to the agent"
    )
    
    inference_parser.add_argument(
        "--use-mock", "--mock",
        action="store_true",
        help="Use mock LLM adapter for testing"
    )
    

def _handle_default_command(args):
    """Handle the default command (no subcommand specified)"""
    # If quick-test is specified, run the quick test
    if hasattr(args, 'quick_test') and args.quick_test:
        return _handle_quick_test(args)
    
    # If load is specified, run with the loaded agent
    if hasattr(args, 'load') and args.load:
        return _handle_loaded_agent(args)
    
    # If eval_command is specified via --eval-command, run the optimizer
    eval_command = None
    if hasattr(args, 'eval_command') and args.eval_command:
        eval_command = args.eval_command
        
    if eval_command:
        print(f"Using evaluation command: {eval_command}")
        # Run the universal optimizer with the eval command
        return _handle_optimize_command(args)
    
    # If no specific action is determined, run the evolve command
    # This makes evolve the default when no subcommand or eval_command is specified
    return _handle_evolve_command(args)

def _handle_quick_test(args):
    """Handle the quick test command"""
    from llm_agent_evolution.quick_test import main as run_quick_test
    
    # Run the quick test
    return run_quick_test(seed=args.seed)

def _handle_loaded_agent(args):
    """Handle running with a loaded agent"""
    # Import necessary modules
    import tomli
    from llm_agent_evolution.domain.model import Agent, Chromosome
    
    # Check if the agent file exists
    if not os.path.exists(args.load):
        print(f"Error: Agent file not found: {args.load}")
        return 1
    
    # Load the agent from the file
    try:
        with open(args.load, 'rb') as f:
            agent_data = tomli.load(f)
        
        # Extract agent information
        agent_info = agent_data.get('agent', {})
        
        # Create the agent
        agent = Agent(
            task_chromosome=Chromosome(
                content=agent_info.get('task_chromosome', {}).get('content', ''),
                type="task"
            ),
            mate_selection_chromosome=Chromosome(
                content=agent_info.get('mate_selection_chromosome', {}).get('content', ''),
                type="mate_selection"
            ),
            mutation_chromosome=Chromosome(
                content=agent_info.get('mutation_chromosome', {}).get('content', ''),
                type="mutation"
            ),
            reward=agent_info.get('reward', 0.0)
        )
        
        # Get the evaluation command
        eval_command = args.eval_command
        if not eval_command:
            print("Error: No evaluation command specified.")
            return 1
        
        # Run the evaluation
        print("=" * 60)
        print("LLM Agent Evolution")
        print("A framework for evolving LLM-based agents")
        print("=" * 60)
        print(f"Loaded agent with ID: {agent_info.get('id', agent.id)}")
        
        # Set up environment for context if provided
        env = os.environ.copy()
        if args.context:
            env['AGENT_CONTEXT'] = args.context
        elif args.context_file and os.path.exists(args.context_file):
            with open(args.context_file, 'r') as f:
                env['AGENT_CONTEXT'] = f.read()
        
        # Run the evaluation command
        import subprocess
        result = subprocess.run(
            eval_command,
            shell=True,
            input=agent.task_chromosome.content,
            text=True,
            capture_output=True,
            env=env
        )
        
        # Extract the reward from the last line of output
        output_lines = result.stdout.strip().split('\n')
        try:
            reward = float(output_lines[-1])
            detailed_output = '\n'.join(output_lines[:-1])
        except (ValueError, IndexError):
            reward = 0.0
            detailed_output = result.stdout
        
        print("\nAgent evaluation complete")
        print(f"Reward: {reward}")
        print(f"\nAgent output: {agent.task_chromosome.content}")
        
        if detailed_output:
            print("\nDetailed evaluation output:")
            print(detailed_output)
        
        return 0
        
    except Exception as e:
        print(f"Error loading or evaluating agent: {e}")
        import traceback
        print(traceback.format_exc())
        return 1

def _handle_evolve_command(args):
    """Handle the evolve command"""
    # Import and run the main application
    from llm_agent_evolution.application import create_application
    
    # Determine the evaluation command
    eval_command = None
    if hasattr(args, 'eval_command') and args.eval_command:
        eval_command = args.eval_command
    
    # Create the application
    app = create_application(
        model_name=args.model,
        log_file=args.log_file,
        use_mock=args.use_mock,
        random_seed=args.seed,
        eval_command=eval_command,
        load_agent_path=args.load if hasattr(args, 'load') else None
    )
    
    # Run the evolution
    app.run_evolution(
        population_size=args.population_size,
        parallel_agents=args.parallel_agents,
        max_evaluations=args.max_evaluations,
        initial_content=args.initial_content if hasattr(args, 'initial_content') else ""
    )
    
    return 0
    
def _handle_optimize_command(args):
    """Handle the optimize command"""
    # Import the universal optimizer
    from llm_agent_evolution.universal_optimize import run_optimizer
    
    # Get initial content from file if specified
    initial_content = ""
    
    # Check for initial content in different argument names based on command
    if hasattr(args, 'optimize_initial_content') and args.optimize_initial_content:
        initial_content = args.optimize_initial_content
    elif hasattr(args, 'initial_content') and args.initial_content:
        initial_content = args.initial_content
    if hasattr(args, 'initial_file') and args.initial_file:
        if not os.path.exists(args.initial_file):
            print(f"Error: Initial content file not found: {args.initial_file}")
            return 1
        with open(args.initial_file, 'r') as f:
            initial_content = f.read()
    
    # Determine evaluation method
    eval_script = None
    if hasattr(args, 'eval_script'):
        eval_script = args.eval_script
    
    eval_command = None
    if hasattr(args, 'eval_command'):
        eval_command = args.eval_command
    
    if not eval_script and not eval_command:
        print("Error: Either eval_command or --eval-script must be specified")
        return 1
        
    # If both are provided, eval_command takes precedence
    if eval_command and not eval_script:
        # Create a temporary script that runs the eval command
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
            eval_script = f.name
            f.write("#!/bin/sh\n")
            f.write(f"{eval_command}\n")
        
        # Make executable
        os.chmod(eval_script, 0o755)
    
    try:
        # Run the optimizer
        script_timeout = args.script_timeout if hasattr(args, 'script_timeout') else 30
        output_format = args.output_format if hasattr(args, 'output_format') else "text"
        max_chars = args.max_chars if hasattr(args, 'max_chars') else 1000
        
        # Ensure we have model_name
        model_name = "openrouter/google/gemini-2.0-flash-001"
        if hasattr(args, 'model'):
            model_name = args.model
        
        # Get population size with fallback
        population_size = 50
        if hasattr(args, 'population_size'):
            population_size = args.population_size
            
        # Get parallel agents with fallback
        parallel_agents = 8
        if hasattr(args, 'parallel_agents'):
            parallel_agents = args.parallel_agents
            
        # Get max evaluations with fallback
        max_evaluations = None
        if hasattr(args, 'max_evaluations'):
            max_evaluations = args.max_evaluations
            
        # Get use_mock with fallback
        use_mock_llm = False
        if hasattr(args, 'use_mock'):
            use_mock_llm = args.use_mock
            
        # Get log file with fallback
        log_file = "universal_optimize.log"
        if hasattr(args, 'log_file'):
            log_file = args.log_file
            
        # Get random seed with fallback
        random_seed = None
        if hasattr(args, 'seed'):
            random_seed = args.seed
            
        # Get output file with fallback
        output_file = None
        if hasattr(args, 'save'):
            output_file = args.save
            
        # Get verbose with fallback
        verbose = False
        if hasattr(args, 'verbose'):
            verbose = args.verbose
        
        # Get context from stdin if available
        context = None
        if hasattr(args, 'context') and args.context:
            context = args.context
        elif hasattr(args, 'context_file') and args.context_file and os.path.exists(args.context_file):
            with open(args.context_file, 'r') as f:
                context = f.read()
        elif not sys.stdin.isatty():
            # Read from stdin if available and not already used for arguments
            try:
                context = sys.stdin.read()
                print(f"Read context from stdin: {context[:50]}{'...' if len(context) > 50 else ''}")
            except Exception as e:
                print(f"Error reading from stdin: {e}")
        
        # Set context in environment if provided
        if context:
            os.environ['AGENT_CONTEXT'] = context
        
        result = run_optimizer(
            eval_script=eval_script,
            population_size=population_size,
            parallel_agents=parallel_agents,
            max_evaluations=max_evaluations,
            use_mock_llm=use_mock_llm,
            model_name=model_name,
            log_file=log_file,
            random_seed=random_seed,
            script_timeout=script_timeout,
            initial_content=initial_content,
            save=output_file,
            output_format=output_format,
            max_chars=max_chars,
            verbose=verbose,
            eval_command=eval_command
        )
        
        return 0
    finally:
        # Clean up temporary script if created
        if eval_command and not eval_script and 'eval_script' in locals() and os.path.exists(eval_script):
            os.remove(eval_script)

def _handle_standalone_command(args):
    """Handle the standalone command"""
    # Import the standalone optimizer
    from llm_agent_evolution.standalone import run_standalone_optimizer
    
    # Run the standalone optimizer
    try:
        # Get initial content
        initial_content = args.standalone_initial_content if hasattr(args, 'standalone_initial_content') else ""
        
        results = run_standalone_optimizer(
            eval_command=args.eval_command,
            population_size=args.population_size,
            parallel_agents=args.parallel_agents,
            max_evaluations=args.max_evaluations,
            initial_content=initial_content,
            random_seed=args.seed,
            verbose=args.verbose if hasattr(args, 'verbose') else False
        )
        
        # Write to save file if specified
        if hasattr(args, 'save') and args.save and results["best_agent"]["content"]:
            with open(args.save, 'w') as f:
                f.write(results["best_agent"]["content"])
            print(f"\nBest result saved to: {args.save}")
            
        return 0
    except Exception as e:
        print(f"Error running standalone optimizer: {e}")
        import traceback
        print(traceback.format_exc())
        return 1

def _handle_demo_command(args):
    """Handle the demo command"""
    # Import the evolution demo
    from llm_agent_evolution.evolution_demo import run_evolution_demo
    
    # Run the demo
    return run_evolution_demo(
        use_mock=args.use_mock,
        initial_content=args.demo_initial_content if hasattr(args, 'demo_initial_content') else "a"
    )

def _handle_inference_command(args):
    """Handle the inference command (run a loaded agent without evolution)"""
    # Import necessary modules
    import tomli
    from llm_agent_evolution.domain.model import Agent, Chromosome
    
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE WITH LOADED AGENT".center(60))
    print("=" * 60)
    
    # Check if the agent file exists
    if not os.path.exists(args.load):
        print(f"Error: Agent file not found: {args.load}")
        return 1
    
    # Load the agent from the file
    try:
        with open(args.load, 'rb') as f:
            agent_data = tomli.load(f)
        
        # Extract agent information
        agent_info = agent_data.get('agent', {})
        
        # Create the agent
        agent = Agent(
            task_chromosome=Chromosome(
                content=agent_info.get('task_chromosome', {}).get('content', ''),
                type="task"
            ),
            mate_selection_chromosome=Chromosome(
                content=agent_info.get('mate_selection_chromosome', {}).get('content', ''),
                type="mate_selection"
            ),
            mutation_chromosome=Chromosome(
                content=agent_info.get('mutation_chromosome', {}).get('content', ''),
                type="mutation"
            ),
            id=agent_info.get('id'),
            reward=agent_info.get('reward', 0.0)
        )
        
        print(f"Loaded agent with ID: {agent.id}")
        print(f"Original reward: {agent.reward}")
        
        # Get context from file or argument
        context = args.context
        if args.context_file and os.path.exists(args.context_file):
            with open(args.context_file, 'r') as f:
                context = f.read()
                print(f"Loaded context from file: {args.context_file}")
        
        # Set up environment for context if provided
        env = os.environ.copy()
        if context:
            env['AGENT_CONTEXT'] = context
            print(f"Using context: {context[:50]}{'...' if len(context) > 50 else ''}")
        
        # Run the evaluation command
        import subprocess
        result = subprocess.run(
            args.eval_command,
            shell=True,
            input=agent.task_chromosome.content,
            text=True,
            capture_output=True,
            env=env
        )
        
        # Extract the reward from the last line of output
        output_lines = result.stdout.strip().split('\n')
        try:
            reward = float(output_lines[-1])
            detailed_output = '\n'.join(output_lines[:-1])
        except (ValueError, IndexError):
            reward = 0.0
            detailed_output = result.stdout
        
        print("\nAgent evaluation complete")
        print(f"Reward: {reward}")
        print(f"\nAgent output: {agent.task_chromosome.content}")
        
        if detailed_output:
            print("\nDetailed evaluation output:")
            print(detailed_output)
        
        return 0
        
    except Exception as e:
        print(f"Error loading or evaluating agent: {e}")
        import traceback
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
