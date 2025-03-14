#!/usr/bin/env python3
"""
Simple example that counts 'a' characters in the first 23 positions.
This is a simplified version of count_a.py for easier testing.
"""
import sys

# Read input from stdin
input_text = sys.stdin.read().strip()

# Count 'a's in the first 23 characters
a_count = input_text[:23].count('a')

# Penalty for exceeding 23 characters
length_penalty = max(0, len(input_text) - 23)

# Calculate reward
reward = a_count - length_penalty

# Print detailed information
print(f"Text: '{input_text[:50]}{'...' if len(input_text) > 50 else ''}'")
print(f"Text length: {len(input_text)}")
print(f"'a' count in first 23 chars: {a_count}")
print(f"Length penalty: {length_penalty}")
print(reward)  # This must be the last line and a number
