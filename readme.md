# Code Intelligence Assistant

> AI-powered code review assistant that learns from high-quality open-source projects to provide context-aware feedback for small development teams.

## 🎯 Problem Statement

Small development teams (2-10 developers) face a dilemma:
- **GitHub Copilot Enterprise** ($40/user/month) is too expensive
- **Free static analysis tools** (Pylint, Ruff) produce noisy, rule-based feedback
- **Generic AI tools** don't understand team-specific patterns and conventions

## 💡 Solution

A code review assistant that:
- ✅ Learns from curated high-quality Python projects (FastAPI, Requests, Pydantic)
- ✅ Provides context-aware suggestions with references to similar code
- ✅ Works with minimal data using few-shot learning (RAG)
- ✅ Shows its reasoning (not a black box)
- ✅ Free and open-source

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
Docker & Docker Compose (optional)
GitHub Personal Access Token
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/code-intelligence-assistant.git
cd code-intelligence-assistant
```

2. **Set up environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Collect training data** (Week 1)
```bash
python scripts/01_collect_github_data.py
```

5. **Run the web interface** (Week 4+)
```bash
streamlit run app/streamlit_app.py
```

## 📊 Project Status

- [x] Week 1: Data collection from GitHub
- [x] Week 2: RAG pipeline foundation
- [ ] Week 3: Intelligence layer
- [ ] Week 4: Streamlit interface
- [ ] Week 5: User testing
- [ ] Week 6: GitHub integration

## 🏗️ Architecture

```
User Input (Code Snippet)
    ↓
Embedding Generation
    ↓
Vector Similarity Search (ChromaDB)
    ↓
Retrieve Similar Examples
    ↓
LLM Review with Context
    ↓
Structured Feedback with References
```

**Key Components:**
- **Data Collection**: GitHub API scraper for PRs and review comments
- **RAG Pipeline**: sentence-transformers + ChromaDB + LangChain
- **LLM**: OpenRouter (free models) or local Ollama
- **Frontend**: Streamlit with interactive UI

## 📚 Learning Dataset

We learn from these curated Python projects:

| Project | Focus Area | Why Selected |
|---------|-----------|--------------|
| FastAPI | API design, async patterns, type hints | Excellent review culture, modern Python |
| Requests | Error handling, API design, UX | Legendary code style, pythonic patterns |
| Pydantic | Data validation, type safety | Detailed reviews, performance focus |

**Data collected**: ~150-200 PR reviews with detailed feedback

## 🎯 Target Users

- Bootcamp graduates building first projects
- Indie developers without team support
- Small startups (2-5 person teams)
- Developers learning Python best practices

## 🔧 Tech Stack

- **Python 3.10+**: Core language
- **LangChain**: RAG orchestration
- **sentence-transformers**: Code embeddings
- **ChromaDB**: Vector database
- **Streamlit**: Web interface
- **GitHub API**: Data collection
- **OpenRouter/Ollama**: LLM inference

## 📈 Success Metrics (Target)

- ✅ 100+ code reviews generated
- ✅ 20+ unique users
- ✅ 60%+ suggestion acceptance rate
- ✅ 5+ user testimonials
- ✅ <5 second review generation time

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome!

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📝 License

MIT License - feel free to use this for learning or building upon

## 👤 Author

**Aishwarya Kandasami**


## 🙏 Acknowledgments

Learning dataset curated from:
- FastAPI by Sebastián Ramírez
- Requests by Kenneth Reitz
- Pydantic by Samuel Colvin

---

**Status**: 🚧 In Development (Week 1/6)
**Last Updated**: October 2025