# config.py
import os

MODEL_CONFIG = {
    'provider': os.getenv('LLM_PROVIDER', 'claude'),
    'model': os.getenv('LLM_MODEL', 'claude-sonnet-4-20250514'),
    'timeout': int(os.getenv('REVIEW_TIMEOUT', '60')),
    'max_tokens': int(os.getenv('MAX_TOKENS', '4000')),
    'hybrid_mode': os.getenv('HYBRID_MODE', 'true').lower() == 'true',  # Add this
}

OLLAMA_MODELS = {
    'fast': 'llama3.1:8b',
    'code': 'qwen2.5-coder:7b',      # BEST for code review
    'deep': 'deepseek-coder:6.7b',
    'large': 'codellama:13b',
}