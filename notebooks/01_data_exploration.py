"""Quick data exploration script."""

import json
from pathlib import Path
from collections import Counter

def analyze_data():
    """Analyze collected PR data."""
    data_dir = Path('data/raw')
    
    total_prs = 0
    total_comments = 0
    comment_lengths = []
    reviewers = Counter()
    files_reviewed = Counter()
    
    print("📊 DATA ANALYSIS REPORT")
    print("="*60)
    
    for repo_dir in data_dir.iterdir():
        if not repo_dir.is_dir():
            continue
        
        print(f"\n📁 {repo_dir.name}")
        print("-"*60)
        
        for json_file in repo_dir.glob('*.json'):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                prs = data.get('prs', [])
                
                total_prs += len(prs)
                print(f"PRs: {len(prs)}")
                
                for pr in prs:
                    comments = pr.get('review_comments', [])
                    total_comments += len(comments)
                    
                    for comment in comments:
                        body = comment.get('body', '')
                        comment_lengths.append(len(body))
                        
                        reviewer = comment.get('author', 'unknown')
                        reviewers[reviewer] += 1
                        
                        file_path = comment.get('path', '')
                        if file_path:
                            files_reviewed[file_path] += 1
                
                print(f"Review Comments: {sum(len(pr.get('review_comments', [])) for pr in prs)}")
    
    print(f"\n{'='*60}")
    print(f"📈 OVERALL STATISTICS")
    print(f"{'='*60}")
    print(f"Total PRs: {total_prs}")
    print(f"Total Review Comments: {total_comments}")
    print(f"Average comments per PR: {total_comments/total_prs:.1f}")
    
    if comment_lengths:
        print(f"\n💬 Comment Length Stats:")
        print(f"  Average: {sum(comment_lengths)/len(comment_lengths):.0f} characters")
        print(f"  Shortest: {min(comment_lengths)} characters")
        print(f"  Longest: {max(comment_lengths)} characters")
    
    print(f"\n👥 Top 10 Reviewers:")
    for reviewer, count in reviewers.most_common(10):
        print(f"  {reviewer}: {count} comments")
    
    print(f"\n📄 Top 10 Most Reviewed Files:")
    for file_path, count in files_reviewed.most_common(10):
        print(f"  {file_path}: {count} comments")
    
    print(f"\n{'='*60}")

if __name__ == '__main__':
    analyze_data()