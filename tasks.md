PLEASE WORK ON THE BELOW ITEMS. NEVER MODIFY THE HEADING! INSTEAD WRITE BELOW EACH HEADING WHAT YOU DID AND IF YOU THINK THE ITEM IS DONE. FOR THE QUESTIONS PLEASE ANSWER THEM AS BEST YOU CAN. LEAVE THE HEADING / THE ITEM ITSELF ALONG! LEAVE THIS SENTENCE IN, DON'T REMOVE IT! USE A SEPARATE SEARCH REPLACE BLOCK FOR EACH HEADING ITEM / TASK SINCE I MIGHT MOVE THEM AROUND AND THEN A BIG SEARCH BLOCK MIGHT NOT MATCH. IF YOU ADD ITEMS DON'T ADD TODO AT THE BEGINNING SINCE YOU SHOULDN'T MODIFY THE HEADING AND I DON'T WANT DONE TODOS TO STILL HAVE THE TODO TEXT

# parallelize pytest 
Added parallel execution for pytest. DONE

# unify entrypoints
Unified CLI interface with single entry point. DONE

# fix errors
Fixed chromosome combination logic to better target length of 23, improved log file creation with fallbacks, and enhanced context passing to evaluation scripts. DONE

# please mark tasks that are done with DONE at the end similar to how its done in the other tasks
Using DONE marker for completed tasks. DONE

# refactor the code
Simplified architecture, reduced duplication, improved error handling. DONE

# please reduce the size of tasks.md
Summarized completed tasks to reduce size. DONE

# remove duplicate code
Consolidated duplicate code between optimizer implementations. DONE

# should we refactor?
Yes - consolidate CLI interface, improve error handling, standardize initialization, reduce duplication.

# what evolution aspects are simplified in our implementation?
1. Fixed mutation rates vs adaptive mutation
2. Simple parent selection without tournaments
3. Basic chromosome combination
4. Limited diversity management
5. No niching/speciation

# how many llm calls do we make during testing in total?
~10-15 LLM calls in integration tests, 3-5 in direct tests. CI environment detection skips real calls. DONE

# reduce readme size a lot
Reduced while maintaining essential information. DONE

# please remove the split between optimize, standalone, .... and unify them and the functionality
Unified with single entry point and shared code patterns. DONE

# loading agents works with toml right? 
Yes, using TOML for agent serialization with --save/--load arguments. DONE

# please add e2e tests for running inference with a loaded toml agent 
Need to implement end-to-end tests for loading agents from TOML files and running inference with them.

# add e2e test for optimizing with stdin input
Need to implement end-to-end tests for optimizing with stdin input as context.

# are we parallelizing evolution for the e2e tests? please do so they are slow
Using ThreadPoolExecutor with configurable --parallel-agents parameter. DONE

