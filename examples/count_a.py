#!/usr/bin/env python3
"""
Example evaluation script that counts 'a' characters in the first 23 positions
and penalizes for exceeding 23 characters.
"""
import sys

def evaluate(text):
    """
    Evaluate the text based on the hidden goal:
    - Reward increases for every 'a' for the first 23 characters
    - Reward decreases for every character after 23 characters
    """
    # Count 'a's in the first 23 characters
    a_count = text[:23].count('a')
    
    # Penalty for exceeding 23 characters
    length_penalty = max(0, len(text) - 23)
    
    # Calculate reward
    reward = a_count - length_penalty
    
    return reward

if __name__ == "__main__":
    # Read input from stdin
    input_text = sys.stdin.read().strip()
    
    # Evaluate the input
    reward = evaluate(input_text)
    
    # Print detailed information for debugging
    print(f"Text: '{input_text[:50]}{'...' if len(input_text) > 50 else ''}'")
    print(f"Text length: {len(input_text)}")
    print(f"'a' count in first 23 chars: {input_text[:23].count('a')}")
    print(f"Length penalty: {max(0, len(input_text) - 23)}")
    print(reward)  # This must be the last line and a number
