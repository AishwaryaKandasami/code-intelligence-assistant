Code Intelligence Assistant
AI-powered code review assistant that learns from high-quality open-source projects to provide context-aware feedback for small development teams.

🎯 The Problem
Small development teams (2-10 developers) face a dilemma:

GitHub Copilot Enterprise ($40/user/month) is too expensive for indie teams
Free static analysis tools (Pylint, Ruff) produce noisy, rule-based feedback
Generic AI tools (ChatGPT) give unverified advice with no references
Manual code reviews are time-consuming and inconsistent
The Real Cost
Without proper code review assistance:

🕐 4+ hours per developer per week on manual reviews
🐛 Bugs slip through that could have been caught
📉 Inconsistent code quality across the team
💸 $80+ per month in wasted developer time using "free" ChatGPT
✨ The Solution
A hybrid AI code review assistant that:

✅ Learns from curated high-quality Python projects (FastAPI, Requests, Pydantic)
✅ Provides verified suggestions with references to actual code (not made up!)
✅ Works with minimal data using RAG (Retrieval-Augmented Generation)
✅ Shows its reasoning - references specific files and line numbers
✅ Hybrid intelligence - Fast local screening + Cloud detailed analysis
✅ Performance optimized - 8.75x faster than initial implementation
✅ Free and open-source - No vendor lock-in

🚀 Quick Start (5 Minutes)
Prerequisites
Required:

Python 3.10+
Ollama Desktop (for local reviews)
OpenRouter API Key (free tier available)
Optional:

8GB+ RAM (for optimal performance)
Installation
bash
# 1. Clone repository
git clone https://github.com/AishwaryaKandasami/code-intelligence-assistant.git
cd code-intelligence-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install streamlit  # For web UI
Configuration
bash
# Create .env file
cat > .env << EOF
# OpenRouter Configuration
OPENROUTER_API_KEY=your_key_here
LLM_MODEL=meta-llama/llama-3.1-8b-instruct

# Ollama Configuration (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b
EOF
Setup Ollama
bash
# 1. Download and install Ollama from ollama.ai
# 2. Pull the code review model
ollama pull qwen2.5-coder:7b

# 3. Verify installation
ollama list
Run the Application
bash
# Start Streamlit UI
streamlit run streamlit_app.py

# Opens automatically at http://localhost:8501
That's it!

First review: ~50s (one-time initialization)
Subsequent reviews: ~30-40s (cached and optimized)
💡 How It Works
Architecture
User Input (Python Code)
↓
Hybrid Strategy Decision
↓
╔═══════════════════════════════════╗
║ Ollama Fast Screening (5-10s)     ║
║ • Qwen2.5-Coder 7B (local)        ║
║ • Quick severity assessment        ║
╚═══════════════════════════════════╝
↓
[If Critical/High Severity Detected]
↓
╔═══════════════════════════════════╗
║ OpenRouter Detailed Review (30s)  ║
║ • Llama 3.1 8B (cloud)            ║
║ • Deep code analysis               ║
╚═══════════════════════════════════╝
↓
╔═══════════════════════════════════╗
║ RAG: Retrieve Similar Examples    ║
║ • Query: numpy vector similarity   ║
║ • Database: 200+ PR reviews from:  ║
║   - FastAPI (API patterns)         ║
║   - Requests (error handling)      ║
║   - Pydantic (validation)          ║
╚═══════════════════════════════════╝
↓
Structured Feedback with References
Key Technical Decisions
Why Hybrid Ollama + OpenRouter?

Ollama: Fast local screening (5-10s), no API costs, privacy
OpenRouter: Detailed analysis (30s), better reasoning for complex issues
Smart routing: Simple code stays local, critical issues get cloud review
Why numpy instead of ChromaDB?

Faster for small datasets (<1000 examples)
No persistence overhead (in-memory)
Simpler architecture for MVP
Can scale to ChromaDB later if needed
Why these specific models?

Qwen2.5-Coder: Best open-source code model at 7B size
Llama 3.1 8B: Strong reasoning with free tier on OpenRouter
📊 Current Status & Metrics
Project Phase: Week 5/6 - User Testing
Last Updated: October 31, 2025

Test Suite Performance
✅ 85.7% Pass Rate (12/14 tests passed)

Category	Tests	Passed	Pass Rate
Bugs	4	4	100%
Security	3	3	100%
Performance	2	2	100%
Style	1	1	100%
Documentation	1	0	0%
Maintainability	1	0	0%
Key Metrics:

⏱️ Average review time: 32.9 seconds
🎯 Issue detection: High accuracy across critical categories
💰 Cost per review: ~$0.001 (nearly free with OpenRouter free tier)
Performance Optimization
🚀 8.75x faster than initial implementation
⚡ First review: 50s (one-time initialization)
⚡ Subsequent reviews: 30-40s (singleton pattern with caching)
⚡ Simple code: 5-10s (Ollama fast path)

Example Review Output
Input Code:

python
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id='{user_id}'"
    return db.execute(query)['data']

API_KEY = "sk-1234567890"
Output:

🔴 CRITICAL - SQL Injection Vulnerability
Location: Line 2
Description: String formatting in SQL query allows injection attacks
Reference: Similar pattern in requests/sessions.py (line 245)
Suggestion: Use parameterized queries

Example Fix:
  query = "SELECT * FROM users WHERE id=?"
  result = db.execute(query, (user_id,))

🔴 CRITICAL - Hardcoded Credentials
Location: Line 5
Description: API keys should never be in source code
Reference: FastAPI environment variable pattern (config.py:45)
Suggestion: Use environment variables

Example Fix:
  import os
  API_KEY = os.getenv('API_KEY')
Notice:

✅ Specific line numbers
✅ Severity ranking (CRITICAL/HIGH/MEDIUM/LOW)
✅ Real references to FastAPI/Requests code (not made up!)
✅ Working code examples
✅ Clear actionable suggestions
🆚 Comparison: Why Not Just Use ChatGPT?
Feature	ChatGPT Free	Claude Free	Code Intelligence Assistant
Verified References	❌ No sources	❌ No sources	✅ Real code from FastAPI/Requests
Response Time	30-60s	20-40s	✅ 5-40s (hybrid)
Structured Output	❌ Plain text	❌ Plain text	✅ Categorized by severity
Code Examples	⚠️ Generic	⚠️ Generic	✅ From actual quality projects
Learning	❌ No memory	❌ No memory	✅ Learned from 200+ PR reviews
Consistency	❌ Varies	❌ Varies	✅ Same standards every time
Export Ready	❌ Copy/paste	❌ Copy/paste	✅ JSON export, one-click
Cost	Free	Free	✅ Free (OpenRouter free tier)
Key Differentiator: We show real code examples you can verify on GitHub. ChatGPT can't do this.

🧪 Testing
Run Test Suite
bash
# Full comprehensive test suite (14 test cases)
python batch_test_all.py

# Single file test
python scripts/hybrid_reviewer_cached.py \
  --file test_cases/security_01_sql_injection.py \
  --output reviews/test_output.json
Test Categories
The test suite covers:

🐛 Bugs: Logic errors, race conditions, mutable defaults, resource leaks
🔒 Security: SQL injection, hardcoded secrets, path traversal
⚡ Performance: O(n²) algorithms, repeated computations
📝 Documentation: Missing docstrings, unclear code
🎨 Style: Type hints, code complexity
🔧 Maintainability: Code organization, technical debt
See TEST_RESULTS_REPORT.md for detailed results.

🎓 Learning Dataset
We learn from 200+ PR reviews from these curated Python projects:

Project	Focus Area	Why Selected	Stars
FastAPI	API design, async patterns, type hints	Excellent review culture, modern Python	75k+
Requests	Error handling, API design, UX	Legendary code style, pythonic patterns	52k+
Pydantic	Data validation, type safety	Detailed reviews, performance focus	20k+
Why these projects?

✅ Active, high-quality code review culture
✅ Modern Python best practices
✅ Well-documented review comments
✅ Production-grade code patterns
✅ Diverse use cases (APIs, libraries, validation)
🎯 Target Users
Perfect for:

🎓 Bootcamp graduates building first projects
💼 Indie developers without team support
🚀 Small startups (2-5 person teams)
📚 Python learners wanting to improve code quality
💰 Teams on a budget (can't afford $40/user/month for Copilot)
Not for:

❌ Large enterprises (use GitHub Copilot Enterprise)
❌ Non-Python projects (coming in future versions)
❌ Teams needing instant (<1s) responses
🛠️ Tech Stack
Backend:

Python 3.10+
Ollama (Qwen2.5-Coder 7B) - Local inference
OpenRouter (Llama 3.1 8B) - Cloud inference
sentence-transformers - Code embeddings
numpy - Vector similarity search
Frontend:

Streamlit - Interactive web UI
Real-time progress updates
Batch processing support
Optimization:

Singleton pattern with lazy loading
@st.cache_resource for component caching
In-process reviews (no subprocess overhead)
ThreadPoolExecutor for batch processing
Infrastructure:

No database required (in-memory)
Can run completely offline (Ollama only mode)
Deploy to Streamlit Cloud (free)

Future Ideas:

Multi-language support (JavaScript, TypeScript, Go, Rust)
Self-hosted enterprise version
Continuous learning from team's PRs
Integration with Slack/Teams for notifications
Historical trend analysis
🤝 Contributing
This is a portfolio project currently in user testing phase.

📄 License
MIT License - See LICENSE file for details

What this means:

✅ Free to use for personal and commercial projects
✅ Modify and distribute freely
✅ Private use allowed
⚠️ No warranty provided
⚠️ Attribution appreciated but not required
👤 Author
Aishwarya Kandasami

GitHub: @AishwaryaKandasami
Project: code-intelligence-assistant
Learning Dataset Credits
This project learns from code reviews in these open-source projects:

FastAPI by Sebastián Ramírez
Requests by Kenneth Reitz
Pydantic by Samuel Colvin



