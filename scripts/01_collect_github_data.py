"""
GitHub Data Collection Script
Collects PRs and review comments from target repositories.

Usage:
    python scripts/01_collect_github_data.py
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from dotenv import load_dotenv
from github import Github, GithubException
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GitHubDataCollector:
    """Collects PR review data from GitHub repositories."""
    
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN not found in environment variables")
        
        # Updated for newer PyGithub version
        from github import Auth
        auth = Auth.Token(self.github_token)
        self.client = Github(auth=auth)
        
        self.data_dir = Path(os.getenv('DATA_DIR', './data'))
        self.raw_dir = self.data_dir / 'raw'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.max_prs_per_repo = int(os.getenv('MAX_PRS_PER_REPO', 100))
        self.min_review_comments = int(os.getenv('MIN_REVIEW_COMMENTS', 3))
        self.days_lookback = int(os.getenv('DAYS_LOOKBACK', 365))
        
        # Fixed rate limit access
        rate_limit = self.client.get_rate_limit()
        remaining = rate_limit.core.remaining if hasattr(rate_limit, 'core') else rate_limit.rate.remaining
        logger.info(f"Initialized GitHub client. Rate limit: {remaining}")
    
    def collect_repo_data(self, repo_name: str) -> Dict:
        """
        Collect PR and review data from a single repository.
        
        Args:
            repo_name: Full repo name (e.g., 'tiangolo/fastapi')
            
        Returns:
            Dictionary with collected data
        """
        logger.info(f"Starting collection for {repo_name}")
        
        try:
            repo = self.client.get_repo(repo_name)
        except GithubException as e:
            logger.error(f"Failed to access repo {repo_name}: {e}")
            return {}
        
        # Calculate date threshold

        from datetime import timezone
        since_date = datetime.now(timezone.utc) - timedelta(days=self.days_lookback)
        
        collected_data = {
            'repo_name': repo_name,
            'collection_date': datetime.now().isoformat(),
            'prs': []
        }
        
        # Fetch merged PRs
        prs = repo.get_pulls(state='closed', sort='updated', direction='desc')
        
        pr_count = 0
        for pr in tqdm(prs, desc=f"Processing {repo_name}", total=self.max_prs_per_repo):
            if pr_count >= self.max_prs_per_repo:
                break
            
            # Filter by date and merge status
            if pr.merged_at and pr.merged_at > since_date:
                pr_data = self._extract_pr_data(pr)
                
                # Only include PRs with meaningful reviews
                if len(pr_data.get('review_comments', [])) >= self.min_review_comments:
                    collected_data['prs'].append(pr_data)
                    pr_count += 1
            
            # Respect rate limits
            self._check_rate_limit()
        
        logger.info(f"Collected {pr_count} PRs from {repo_name}")
        
        # Save to file
        self._save_data(repo_name, collected_data)
        
        return collected_data
    
    def _extract_pr_data(self, pr) -> Dict:
        """Extract relevant data from a PR."""
        pr_data = {
            'number': pr.number,
            'title': pr.title,
            'description': pr.body or "",
            'author': pr.user.login if pr.user else "unknown",
            'created_at': pr.created_at.isoformat(),
            'merged_at': pr.merged_at.isoformat() if pr.merged_at else None,
            'url': pr.html_url,
            'files_changed': [],
            'review_comments': [],
            'general_comments': []
        }
        
        # Get changed files (limit to Python files)
        try:
            files = pr.get_files()
            for file in files:
                if file.filename.endswith('.py'):
                    pr_data['files_changed'].append({
                        'filename': file.filename,
                        'status': file.status,
                        'additions': file.additions,
                        'deletions': file.deletions,
                        'patch': file.patch if hasattr(file, 'patch') else None
                    })
        except GithubException as e:
            logger.warning(f"Could not fetch files for PR #{pr.number}: {e}")
        
        # Get review comments (inline code comments)
        try:
            review_comments = pr.get_review_comments()
            for comment in review_comments:
                pr_data['review_comments'].append({
                    'author': comment.user.login if comment.user else "unknown",
                    'body': comment.body,
                    'path': comment.path,
                    'line': comment.original_line if hasattr(comment, 'original_line') else None,
                    'created_at': comment.created_at.isoformat(),
                    'diff_hunk': comment.diff_hunk if hasattr(comment, 'diff_hunk') else None
                })
        except GithubException as e:
            logger.warning(f"Could not fetch review comments for PR #{pr.number}: {e}")
        
        # Get general PR comments
        try:
            issue_comments = pr.get_issue_comments()
            for comment in issue_comments:
                pr_data['general_comments'].append({
                    'author': comment.user.login if comment.user else "unknown",
                    'body': comment.body,
                    'created_at': comment.created_at.isoformat()
                })
        except GithubException as e:
            logger.warning(f"Could not fetch issue comments for PR #{pr.number}: {e}")
        
        return pr_data
    
    def _check_rate_limit(self):
        """Check and handle GitHub API rate limits."""
        rate_limit = self.client.get_rate_limit()
        
        # Handle both old and new PyGithub API
        if hasattr(rate_limit, 'core'):
            remaining = rate_limit.core.remaining
            reset_time = rate_limit.core.reset
        else:
            remaining = rate_limit.rate.remaining
            reset_time = rate_limit.rate.reset
        
        if remaining < 100:
            sleep_time = (reset_time - datetime.now()).total_seconds() + 10
            logger.warning(f"Rate limit low ({remaining}). Sleeping for {sleep_time:.0f}s")
            time.sleep(max(sleep_time, 0))
    
    def _save_data(self, repo_name: str, data: Dict):
        """Save collected data to JSON file."""
        # Create repo-specific directory
        repo_dir = self.raw_dir / repo_name.replace('/', '_')
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with timestamp
        filename = f"prs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = repo_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved data to {filepath}")
    
    def collect_all_repos(self, repo_list: Optional[List[str]] = None):
        """Collect data from all configured repositories."""
        if repo_list is None:
            repo_list = os.getenv('TARGET_REPOS', '').split(',')
            repo_list = [r.strip() for r in repo_list if r.strip()]
        
        if not repo_list:
            logger.error("No repositories configured. Set TARGET_REPOS in .env")
            return
        
        logger.info(f"Will collect data from {len(repo_list)} repositories")
        
        results = {}
        for repo_name in repo_list:
            try:
                results[repo_name] = self.collect_repo_data(repo_name)
                # Be nice to GitHub API
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error collecting data from {repo_name}: {e}")
                results[repo_name] = {'error': str(e)}
        
        # Summary
        total_prs = sum(len(data.get('prs', [])) for data in results.values())
        logger.info(f"\n{'='*50}")
        logger.info(f"Collection Complete!")
        logger.info(f"Total PRs collected: {total_prs}")
        for repo, data in results.items():
            if 'error' not in data:
                logger.info(f"  {repo}: {len(data.get('prs', []))} PRs")
        logger.info(f"{'='*50}\n")
        
        return results


def main():
    """Main execution function."""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║       Code Intelligence Assistant - Data Collection        ║
    ║                      Week 1: GitHub Scraper                ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Verify environment setup
    if not os.getenv('GITHUB_TOKEN'):
        print("❌ ERROR: GITHUB_TOKEN not found!")
        print("Please:")
        print("1. Copy .env.example to .env")
        print("2. Add your GitHub Personal Access Token")
        print("3. Get token from: https://github.com/settings/tokens")
        return
    
    # Initialize collector
    try:
        collector = GitHubDataCollector()
    except Exception as e:
        logger.error(f"Failed to initialize collector: {e}")
        return
    
    # Collect data
    try:
        collector.collect_all_repos()
        print("\n✅ Data collection complete!")
        print(f"📁 Check the ./data/raw/ directory for collected data")
    except KeyboardInterrupt:
        logger.info("\n⚠️  Collection interrupted by user")
    except Exception as e:
        logger.error(f"❌ Collection failed: {e}")


if __name__ == "__main__":
    main()