#!/usr/bin/env python3
"""
Example evaluation script that evaluates writing quality.
"""
import sys
import re

def count_sentences(text):
    """Count the number of sentences in the text"""
    # Simple sentence detection using regex
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

def count_words(text):
    """Count the number of words in the text"""
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def average_word_length(text):
    """Calculate the average word length"""
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0
    return sum(len(word) for word in words) / len(words)

def evaluate_writing(text):
    """Evaluate writing quality"""
    # Count basic metrics
    num_sentences = count_sentences(text)
    num_words = count_words(text)
    avg_word_len = average_word_length(text)
    
    # Check for minimum content
    if num_words < 5:
        return 0
    
    # Calculate words per sentence
    words_per_sentence = num_words / max(1, num_sentences)
    
    # Penalize very long or very short sentences
    sentence_length_score = 10 - abs(words_per_sentence - 15)
    sentence_length_score = max(0, sentence_length_score) / 10
    
    # Penalize very long or very short words
    word_length_score = 10 - abs(avg_word_len - 5) * 2
    word_length_score = max(0, word_length_score) / 10
    
    # Check for variety of sentence beginnings
    first_words = []
    for sentence in re.split(r'[.!?]+', text):
        sentence = sentence.strip()
        if sentence:
            words = re.findall(r'\b\w+\b', sentence)
            if words:
                first_words.append(words[0].lower())
    
    unique_beginnings = len(set(first_words)) / max(1, len(first_words))
    
    # Calculate final score
    base_score = 5
    base_score += sentence_length_score * 2
    base_score += word_length_score * 2
    base_score += unique_beginnings * 3
    
    # Bonus for longer texts (up to a point)
    length_bonus = min(3, num_words / 100)
    
    return base_score + length_bonus

if __name__ == "__main__":
    # Read input from stdin
    text = sys.stdin.read()
    
    # Calculate metrics
    num_sentences = count_sentences(text)
    num_words = count_words(text)
    avg_word_len = average_word_length(text)
    
    # Print metrics
    print(f"Number of sentences: {num_sentences}")
    print(f"Number of words: {num_words}")
    print(f"Average word length: {avg_word_len:.2f}")
    
    # Calculate and print reward
    reward = evaluate_writing(text)
    print(f"Writing quality score: {reward:.2f}")
    print(reward)  # Last line must be the reward
