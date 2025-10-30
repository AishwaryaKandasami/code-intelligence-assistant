# Create scripts/03_analyze_output.py
"""Analyze processed output quality."""
import json
from pathlib import Path
from collections import Counter

def analyze_output():
    processed_file = Path('data/processed/reviews.jsonl')
    
    examples = []
    with open(processed_file, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Total examples: {len(examples)}")
    print(f"\nSample example:")
    print(json.dumps(examples[0], indent=2))
    
    # Analyze patterns
    repos = Counter(ex['repo_name'] for ex in examples)
    reviewers = Counter(ex['reviewer'] for ex in examples)
    
    print(f"\nExamples by repo:")
    for repo, count in repos.most_common():
        print(f"  {repo}: {count}")
    
    print(f"\nTop 5 reviewers:")
    for reviewer, count in reviewers.most_common(5):
        print(f"  {reviewer}: {count}")

if __name__ == '__main__':
    analyze_output()