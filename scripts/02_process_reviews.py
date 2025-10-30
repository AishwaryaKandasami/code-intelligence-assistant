"""
Data Processing Script - Week 1 Day 2
Transforms raw GitHub PR data into clean code review examples.
"""

import json
from pathlib import Path
from typing import List, Dict
from collections import Counter
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewDataProcessor:
    """Processes raw PR data into training examples."""
    
    def __init__(self):
        self.raw_dir = Path('data/raw')
        self.processed_dir = Path('data/processed')
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality filters (updated based on your data analysis)
        self.min_comment_length = 20     # characters (filter "ok", "lgtm")
        self.max_comment_length = 2000   # characters (very long comments)
        self.min_code_length = 10        # characters
        
        # Patterns to exclude
        self.exclude_patterns = [
            'lgtm', 'looks good to me', 'looks good',
            '👍', '✅', '🎉', 'thanks', 'thank you',
            'merged', 'done', '+1', 'ok'
        ]
        
        # Track statistics
        self.stats = {
            'total_prs': 0,
            'total_comments': 0,
            'filtered_comments': 0,
            'examples_created': 0,
            'no_code_context': 0,
            'too_short': 0,
            'too_long': 0,
            'meaningless': 0,
	    'ping_or_response': 0 
        }

	 # Add these new patterns
        self.ping_patterns = [
            r'^@\w+',  # Starts with mention
            r'please check',
            r'pushed.*commit',
            r'updated commit',
            r'ptal',
            r'i hope',
            r'looks better now',
        ]
        
        self.min_words = 5  # At least 5 words
    
    def is_ping_or_response(self, comment: str) -> bool:
        """Filter pings and conversational responses."""
        comment_lower = comment.lower()
        
        for pattern in self.ping_patterns:
            if re.search(pattern, comment_lower):
                return True
        
        # Check word count
        words = comment.split()
        if len(words) < self.min_words:
            return True
        
        # Check emoji density
        emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF]', comment))
        if emoji_count > 2:
            return True
        
        return False
    
    def is_meaningful_comment(self, comment: str) -> bool:
        """Enhanced quality check."""
        # Existing checks
        if len(comment) < self.min_comment_length:
            self.stats['too_short'] += 1
            return False
        
        if len(comment) > self.max_comment_length:
            self.stats['too_long'] += 1
            return False
        
        # NEW: Check for pings/responses
        if self.is_ping_or_response(comment):
            self.stats['ping_or_response'] += 1
            return False
        
        # Existing exclude patterns
        comment_lower = comment.lower().strip()
        for pattern in self.exclude_patterns:
            if comment_lower == pattern or comment_lower.startswith(pattern + ' '):
                self.stats['meaningless'] += 1
                return False
        
        return True
    
    def load_raw_data(self) -> List[Dict]:
        """Load all JSON files from raw directory."""
        logger.info("Loading raw data...")
        all_data = []
        
        for repo_dir in self.raw_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            
            for json_file in repo_dir.glob('*.json'):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.append(data)
                    logger.info(f"Loaded {len(data.get('prs', []))} PRs from {repo_dir.name}")
        
        return all_data
    
     
    def extract_examples(self, raw_data: List[Dict]) -> List[Dict]:
        """Extract code review examples from raw data."""
        examples = []
        example_id = 0
        
        logger.info("Extracting code review examples...")
        
        for data in raw_data:
            repo_name = data.get('repo_name', 'unknown')
            prs = data.get('prs', [])
            self.stats['total_prs'] += len(prs)
            
            for pr in prs:
                pr_number = pr.get('number')
                pr_title = pr.get('title', '')
                review_comments = pr.get('review_comments', [])
                
                self.stats['total_comments'] += len(review_comments)
                
                for comment in review_comments:
                    comment_body = comment.get('body', '').strip()
                    
                    # Check if comment is meaningful
                    if not self.is_meaningful_comment(comment_body):
                        self.stats['filtered_comments'] += 1
                        continue
                    
                    # Check if we have code context
                    diff_hunk = comment.get('diff_hunk', '')
                    if not diff_hunk or len(diff_hunk) < self.min_code_length:
                        self.stats['no_code_context'] += 1
                        self.stats['filtered_comments'] += 1
                        continue
                    
                    # Create example
                    example = {
                        'id': f"{repo_name.replace('/', '_')}_pr{pr_number}_comment{example_id}",
                        'repo_name': repo_name,
                        'pr_number': pr_number,
                        'pr_title': pr_title,
                        'file_path': comment.get('path', ''),
                        'line_number': comment.get('line'),
                        'code_snippet': diff_hunk,
                        'review_comment': comment_body,
                        'reviewer': comment.get('author', 'unknown'),
                        'created_at': comment.get('created_at', '')
                    }
                    
                    examples.append(example)
                    example_id += 1
                    self.stats['examples_created'] += 1
        
        logger.info(f"Extracted {len(examples)} examples from {self.stats['total_prs']} PRs")
        return examples
    
    def save_processed_data(self, examples: List[Dict]):
        """Save processed examples as JSONL."""
        output_file = self.processed_dir / 'reviews.jsonl'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_file}")
    
    def print_statistics(self):
        """Print processing statistics."""
        logger.info("\n" + "="*50)
        logger.info("PROCESSING STATISTICS")
        logger.info("="*50)
        logger.info(f"Total PRs processed: {self.stats['total_prs']}")
        logger.info(f"Total comments found: {self.stats['total_comments']}")
        logger.info(f"Filtered out: {self.stats['filtered_comments']}")
        logger.info(f"  - Too short: {self.stats['too_short']}")
        logger.info(f"  - Too long: {self.stats['too_long']}")
        logger.info(f"  - Meaningless: {self.stats['meaningless']}")
        logger.info(f"  - Ping/response: {self.stats['ping_or_response']}")
        logger.info(f"  - No code context: {self.stats['no_code_context']}")
        logger.info(f"Examples created: {self.stats['examples_created']}")
        
        if self.stats['total_comments'] > 0:
            acceptance_rate = (self.stats['examples_created'] / self.stats['total_comments']) * 100
            logger.info(f"Acceptance rate: {acceptance_rate:.1f}%")
        
        logger.info("="*50)


def main():
    """Main processing function."""
    processor = ReviewDataProcessor()
    
    # Load raw data
    raw_data = processor.load_raw_data()
    
    # Extract examples
    examples = processor.extract_examples(raw_data)
    
    # Save processed data
    if examples:
        processor.save_processed_data(examples)
    else:
        logger.warning("No examples extracted!")
    
    # Print statistics
    processor.print_statistics()


if __name__ == '__main__':
    main()