"""Quick quality check on processed data."""

import json
import random
from pathlib import Path

def quality_check():
    """Check random examples for quality."""
    
    file = Path('data/processed/reviews.jsonl')
    examples = [json.loads(line) for line in open(file, encoding='utf-8')]
    
    print('='*80)
    print(f'QUALITY CHECK: Random 10 Examples from {len(examples)} total')
    print('='*80)
    print()
    
    for i, ex in enumerate(random.sample(examples, 10), 1):
        print(f'{i}. REVIEWER: {ex["reviewer"]}')
        print(f'   REPO: {ex["repo_name"]}')
        print(f'   FILE: {ex["file_path"]}')
        print(f'   COMMENT: {ex["review_comment"][:100]}...')
        print()
    
    # Statistics
    print('='*80)
    print('STATISTICS:')
    print('='*80)
    
    from collections import Counter
    
    reviewers = Counter(ex['reviewer'] for ex in examples)
    repos = Counter(ex['repo_name'] for ex in examples)
    
    print(f'\nTotal examples: {len(examples)}')
    print(f'\nTop 5 reviewers:')
    for reviewer, count in reviewers.most_common(5):
        print(f'  {reviewer}: {count} comments')
    
    print(f'\nExamples by repo:')
    for repo, count in repos.most_common():
        print(f'  {repo}: {count} examples')
    
    # Average comment length
    avg_length = sum(len(ex['review_comment']) for ex in examples) / len(examples)
    print(f'\nAverage comment length: {avg_length:.0f} characters')
    
    print()
    print('='*80)

if __name__ == '__main__':
    quality_check()