# Quick Start Guide - Week 1

Get your Code Intelligence Assistant up and running in 30 minutes!

## Prerequisites

- ✅ Python 3.10 or higher
- ✅ Git
- ✅ GitHub account
- ✅ 2 Python files already in your GitHub repo

## Step 1: Clone & Setup (5 minutes)

```bash
# Clone your repository
git clone https://github.com/yourusername/code-intelligence-assistant.git
cd code-intelligence-assistant

# Run setup script (creates venv, installs packages)
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Step 2: Get GitHub Token (5 minutes)

1. Go to https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Give it a name: "Code Intelligence Assistant"
4. Select scopes:
   - ✅ `repo` (all checkboxes)
   - ✅ `read:org`
5. Click **"Generate token"**
6. Copy the token (you won't see it again!)

## Step 3: Configure Environment (2 minutes)

```bash
# Edit .env file
nano .env  # or use your favorite editor

# Add your GitHub token:
GITHUB_TOKEN=ghp_your_actual_token_here

# Configure target repos (these are the defaults):
TARGET_REPOS=tiangolo/fastapi,psf/requests,pydantic/pydantic

# Optional: Add LLM API key (can do later)
# OPENAI_API_KEY=sk-...
# or
# ANTHROPIC_API_KEY=sk-ant-...

# Save and exit
```

## Step 4: Test Your Setup (2 minutes)

```bash
# Quick test
python3 -c "from src.config import settings; print(f'✅ Config loaded! Target repos: {settings.target_repos}')"

# Should output:
# ✅ Config loaded! Target repos: ['tiangolo/fastapi', 'psf/requests', 'pydantic/pydantic']
```

## Step 5: Collect Your First Data (15 minutes)

```bash
# Start data collection
python scripts/01_collect_github_data.py

# This will:
# - Connect to GitHub API
# - Fetch last 100 PRs from each repo
# - Filter for PRs with meaningful reviews
# - Save to data/raw/
```

**What to expect:**
```
╔════════════════════════════════════════════════════════════╗
║       Code Intelligence Assistant - Data Collection        ║
║                      Week 1: GitHub Scraper                ║
╚════════════════════════════════════════════════════════════╝

INFO - Initialized GitHub client. Rate limit: 4997
INFO - Starting collection for tiangolo/fastapi
Processing tiangolo/fastapi: 100%|████████| 100/100 [02:15<00:00]
INFO - Collected 47 PRs from tiangolo/fastapi
...
```

## Step 6: Verify Your Data (1 minute)

```bash
# Check collected data
ls -la data/raw/

# Should see directories like:
# tiangolo_fastapi/
# psf_requests/
# pydantic_pydantic/

# Look at one example
cat data/raw/tiangolo_fastapi/prs_*.json | head -50
```

## 🎉 Success! What You Have Now

✅ Working development environment  
✅ GitHub data collection pipeline  
✅ 100-200 real PR reviews from top Python projects  
✅ Structured data ready for processing  

## 📊 Quick Stats

Check what you collected:

```bash
# Run a quick analysis
python -c "
import json
from pathlib import Path

for repo_dir in Path('data/raw').iterdir():
    if repo_dir.is_dir():
        for file in repo_dir.glob('*.json'):
            data = json.load(open(file))
            prs = len(data.get('prs', []))
            comments = sum(len(pr.get('review_comments', [])) for pr in data.get('prs', []))
            print(f'{repo_dir.name}: {prs} PRs, {comments} review comments')
"
```

## 🚨 Common Issues & Solutions

### Issue: "Rate limit exceeded"
**Solution:** GitHub API has limits (5000/hour). The script handles this automatically, just wait.

### Issue: "No module named 'github'"
**Solution:** Make sure virtual environment is activated:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "GITHUB_TOKEN not found"
**Solution:** Check your .env file exists and has the token:
```bash
cat .env | grep GITHUB_TOKEN
```

### Issue: "Collecting 0 PRs"
**Solution:** Repo might not have many recent merged PRs. Try adjusting in .env:
```bash
DAYS_LOOKBACK=730  # Look back 2 years instead of 1
MIN_REVIEW_COMMENTS=2  # Lower threshold
```

## 🎯 Next Steps (Week 1 Remaining Days)

**Day 2-3:** Process the collected data
```bash
python scripts/02_process_reviews.py
```

**Day 4-5:** Create embeddings
```bash
python scripts/03_create_embeddings.py
```

**Day 6-7:** Test retrieval
```bash
python scripts/04_test_retrieval.py
```

## 📚 Useful Commands

```bash
# Reactivate environment (after closing terminal)
source venv/bin/activate

# Update dependencies
pip install -r requirements.txt --upgrade

# Check logs
tail -f logs/collection.log

# Clean data and restart
rm -rf data/raw/*
python scripts/01_collect_github_data.py
```

## 🆘 Need Help?

- Check `README.md` for full documentation
- Review `docs/architecture.md` for system design
- Look at example data in `data/raw/`
- Run with debug mode: `export LOG_LEVEL=DEBUG`

## 💡 Pro Tips

1. **Start small**: Collect from 1 repo first to test
   ```bash
   # In .env, set:
   TARGET_REPOS=tiangolo/fastapi
   MAX_PRS_PER_REPO=20
   ```

2. **Monitor rate limits**: 
   ```bash
   python -c "from github import Github; g = Github('your_token'); print(g.get_rate_limit())"
   ```

3. **Save your progress**: Commit often!
   ```bash
   git add .
   git commit -m "Week 1: Collected data from GitHub"
   git push
   ```

---

**Time to completion:** ~30 minutes  
**Data collected:** 100-200 PR reviews  
**Ready for:** Week 2 - RAG Pipeline  

Let's build something awesome! 🚀