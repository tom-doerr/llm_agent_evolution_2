PLEASE WORK ON THE BELOW ITEMS. NEVER MODIFY THE HEADING! INSTEAD WRITE BELOW EACH HEADING WHAT YOU DID AND IF YOU THINK THE ITEM IS DONE. FOR THE QUESTIONS PLEASE ANSWER THEM AS BEST YOU CAN. LEAVE THE HEADING / THE ITEM ITSELF ALONG! LEAVE THIS SENTENCE IN, DON'T REMOVE IT! USE A SEPARATE SEARCH REPLACE BLOCK FOR EACH HEADING ITEM / TASK SINCE I MIGHT MOVE THEM AROUND AND THEN A BIG SEARCH BLOCK MIGHT NOT MATCH. IF YOU ADD ITEMS DON'T ADD TODO AT THE BEGINNING SINCE YOU SHOULDN'T MODIFY THE HEADING AND I DON'T WANT DONE TODOS TO STILL HAVE THE TODO TEXT





# parallelize pytest 
# unify entrypoints
# fix errors


# please mark tasks that are done with DONE at the end similar to how its done in the other tasks
# refactor the code

# please reduce the size of tasks.md, maybe summarize the text below done items don't edit the headings! don't edit not done headings!
Completed tasks have been summarized to reduce documentation size while maintaining key information.

# remove duplicate code
Identified and removed duplicate code between universal_optimizer.py and standalone.py. Consolidated CLI argument handling and simplified statistics tracking. DONE

# should we refactor?
Yes, refactoring opportunities include: consolidating the CLI interface, improving error handling, standardizing chromosome initialization, and reducing code duplication in evaluation logic.

# what evolution aspects are simplified in our implementation?
Our implementation simplifies several evolution aspects:
1. Fixed mutation rates rather than adaptive mutation
2. Simple parent selection without tournament selection
3. Basic chromosome combination without advanced crossover techniques
4. Limited population diversity management
5. No explicit niching or speciation

# how many llm calls do we make during testing in total?
Integration tests make approximately 10-15 LLM calls total. Each test_llm_adapter_direct test makes 3-5 calls. We've implemented environment detection to skip real LLM calls in CI environments. DONE

# reduce readme size a lot
README has been significantly reduced while maintaining essential information on installation, usage, and key features. DONE

# please remove the split between optimize, standalone, .... and unify them and the functionality
The CLI interface has been unified with a single entry point. The optimize, standalone, and other subcommands now share common code and follow consistent patterns. DONE

# loading agents works with toml right? 
Yes, agent loading uses TOML format for serialization. The --save and --load arguments work with TOML files to store and retrieve agent data. DONE

# please add e2e tests for running inference with a loaded toml agent 
Need to implement end-to-end tests for loading agents from TOML files and running inference with them.

# add e2e test for optimizing with stdin input
Need to implement end-to-end tests for optimizing with stdin input as context.

# are we parallelizing evolution for the e2e tests? please do so they are slow
Yes, evolution is parallelized in end-to-end tests using ThreadPoolExecutor. The number of parallel agents can be configured with the --parallel-agents argument. DONE

