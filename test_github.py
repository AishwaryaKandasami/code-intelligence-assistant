from github import Github
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("GITHUB_TOKEN")
g = Github(token)  # Deprecation warning is okay for now

# Get rate limits
rate = g.get_rate_limit()

print(f"GitHub connected! Remaining requests: {rate.core.remaining}/{rate.core.limit}")

