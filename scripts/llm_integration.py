"""
LLM Integration - Week 2
Multi-provider support: OpenRouter, Ollama, OpenAI

Implements:
- Provider abstraction
- Retry logic with exponential backoff
- Streaming support
- Token counting and cost tracking
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Generator, List
from dataclasses import dataclass
from enum import Enum
import requests
from dotenv import load_dotenv
import json

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: LLMProvider
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class LLMResponse:
    """LLM response with metadata."""
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    latency: float


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        stream: bool = False
    ) -> LLMResponse:
        """Generate completion from prompt."""
        pass
    
    @abstractmethod
    def stream_generate(
        self,
        prompt: str
    ) -> Generator[str, None, None]:
        """Stream completion tokens."""
        pass
    
    def _retry_with_backoff(
        self,
        func,
        max_retries: int = 3,
        initial_delay: float = 1.0
    ):
        """Retry with exponential backoff."""
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter provider - access to multiple models.
    
    Free tier models:
    - google/gemma-2-9b-it:free
    - meta-llama/llama-3.1-8b-instruct:free
    - mistralai/mistral-7b-instruct:free
    """
    
    COST_PER_1M_TOKENS = {
        'google/gemma-2-9b-it:free': 0.0,
        'meta-llama/llama-3.1-8b-instruct:free': 0.0,
        'mistralai/mistral-7b-instruct:free': 0.0,
        'anthropic/claude-3-haiku': 0.25,
        'openai/gpt-4o-mini': 0.15,
    }
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def generate(
        self,
        prompt: str,
        stream: bool = False
    ) -> LLMResponse:
        """Generate completion via OpenRouter."""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/AishwaryaKandasami/code-intelligence-assistant",
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        def _make_request():
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        
        data = self._retry_with_backoff(_make_request)
        
        # Parse response
        content = data['choices'][0]['message']['content']
        usage = data.get('usage', {})
        
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
        
        # Calculate cost
        cost_per_1m = self.COST_PER_1M_TOKENS.get(self.config.model, 0)
        cost = (total_tokens / 1_000_000) * cost_per_1m
        
        latency = time.time() - start_time
        
        return LLMResponse(
            content=content,
            model=self.config.model,
            provider='openrouter',
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            latency=latency
        )
    
    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        """Stream completion tokens."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk['choices'][0]['delta'].get('content', '')
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider - local LLM execution.
    
    Recommended models:
    - codellama:7b-instruct
    - llama3.1:8b
    - mistral:7b-instruct
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
    
    def generate(
        self,
        prompt: str,
        stream: bool = False
    ) -> LLMResponse:
        """Generate completion via Ollama."""
        start_time = time.time()
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }
        
        def _make_request():
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        
        data = self._retry_with_backoff(_make_request)
        
        content = data['response']
        
        # Ollama provides token counts
        prompt_tokens = data.get('prompt_eval_count', 0)
        completion_tokens = data.get('eval_count', 0)
        total_tokens = prompt_tokens + completion_tokens
        
        latency = time.time() - start_time
        
        return LLMResponse(
            content=content,
            model=self.config.model,
            provider='ollama',
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=0.0,  # Local is free
            latency=latency
        )
    
    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        """Stream completion tokens."""
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                content = data.get('response', '')
                if content:
                    yield content
                if data.get('done', False):
                    break


class LLMFactory:
    """Factory for creating LLM providers."""
    
    PROVIDER_MAP = {
        LLMProvider.OPENROUTER: OpenRouterProvider,
        LLMProvider.OLLAMA: OllamaProvider,
    }
    
    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLMProvider:
        """Create LLM provider from config."""
        provider_class = cls.PROVIDER_MAP.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        return provider_class(config)
    
    @classmethod
    def create_from_env(cls) -> BaseLLMProvider:
        """Create provider from environment variables."""
        provider_name = os.getenv('LLM_PROVIDER', 'openrouter').lower()
        model = os.getenv('LLM_MODEL', 'google/gemma-2-9b-it:free')
        
        try:
            provider = LLMProvider(provider_name)
        except ValueError:
            logger.warning(f"Unknown provider '{provider_name}', defaulting to OpenRouter")
            provider = LLMProvider.OPENROUTER
        
        config = LLMConfig(
            provider=provider,
            model=model,
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '2000'))
        )
        
        return cls.create(config)


class CodeReviewLLM:
    """
    High-level interface for code review generation.
    Handles prompt formatting, response parsing, and error handling.
    """
    
    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.provider = provider or LLMFactory.create_from_env()
    
    def generate_review(
        self,
        prompt: str,
        stream: bool = False
    ) -> LLMResponse:
        """
        Generate code review from RAG prompt.
        
        Args:
            prompt: Complete prompt from RAG pipeline
            stream: Whether to stream response
            
        Returns:
            LLMResponse with review content
        """
        logger.info("="*60)
        logger.info(f"LLM GENERATION - {self.provider.config.provider.value}")
        logger.info("="*60)
        logger.info(f"Model: {self.provider.config.model}")
        logger.info(f"Temperature: {self.provider.config.temperature}")
        logger.info(f"Max tokens: {self.provider.config.max_tokens}")
        logger.info("-"*60)
        
        try:
            if stream:
                logger.info("Streaming generation...")
                content_parts = []
                for token in self.provider.stream_generate(prompt):
                    print(token, end='', flush=True)
                    content_parts.append(token)
                print()  # Newline after streaming
                
                # Create response object (approximate tokens)
                content = ''.join(content_parts)
                return LLMResponse(
                    content=content,
                    model=self.provider.config.model,
                    provider=self.provider.config.provider.value,
                    prompt_tokens=len(prompt) // 4,
                    completion_tokens=len(content) // 4,
                    total_tokens=(len(prompt) + len(content)) // 4,
                    cost=0.0,
                    latency=0.0
                )
            else:
                logger.info("Generating review...")
                response = self.provider.generate(prompt, stream=False)
                
                logger.info("-"*60)
                logger.info(f"✅ Generation complete!")
                logger.info(f"Tokens: {response.total_tokens} (prompt: {response.prompt_tokens}, completion: {response.completion_tokens})")
                logger.info(f"Latency: {response.latency:.2f}s")
                logger.info(f"Cost: ${response.cost:.4f}")
                logger.info("="*60)
                
                return response
                
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            raise
    
    def parse_review_response(self, response: LLMResponse) -> Dict:
        """
        Parse structured review from LLM response.
        
        Extracts:
        - Summary
        - Issues (categorized)
        - Suggestions
        - Positive aspects
        
        Args:
            response: LLM response
            
        Returns:
            Parsed review dict
        """
        content = response.content
        
        parsed = {
            'summary': '',
            'issues': [],
            'suggestions': [],
            'positive_aspects': [],
            'raw_content': content,
            'metadata': {
                'model': response.model,
                'provider': response.provider,
                'tokens': response.total_tokens,
                'cost': response.cost,
                'latency': response.latency
            }
        }
        
        # Simple parsing (you can enhance with regex)
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect sections
            if '## summary' in line_lower:
                current_section = 'summary'
            elif '## issues' in line_lower or '## problems' in line_lower:
                current_section = 'issues'
            elif '## suggestions' in line_lower or '## recommendations' in line_lower:
                current_section = 'suggestions'
            elif '## positive' in line_lower:
                current_section = 'positive_aspects'
            elif line.startswith('###') and current_section == 'issues':
                # Issue title
                parsed['issues'].append({'title': line.strip('#').strip(), 'details': []})
            elif line.startswith('-') or line.startswith('*'):
                # Bullet point
                if current_section == 'suggestions':
                    parsed['suggestions'].append(line.strip('- *').strip())
                elif current_section == 'positive_aspects':
                    parsed['positive_aspects'].append(line.strip('- *').strip())
            elif current_section and line.strip():
                # Content for current section
                if current_section == 'summary':
                    parsed['summary'] += line + ' '
                elif current_section == 'issues' and parsed['issues']:
                    parsed['issues'][-1]['details'].append(line.strip())
        
        parsed['summary'] = parsed['summary'].strip()
        
        return parsed


def demo_providers():
    """Demonstrate different LLM providers."""
    print("""
╔══════════════════════════════════════════════════════════╗
║     LLM Integration - Multi-Provider Demo               ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    sample_prompt = """You are a code reviewer. Review this code:

```python
def fetch_data(user_id):
    result = db.query(f"SELECT * FROM users WHERE id = {user_id}")
    return result[0]
```

Provide brief feedback on potential issues."""
    
    # Test OpenRouter (if configured)
    if os.getenv('OPENROUTER_API_KEY'):
        print("\n🌐 Testing OpenRouter Provider...")
        print("-"*60)
        
        try:
            config = LLMConfig(
                provider=LLMProvider.OPENROUTER,
                model='google/gemma-2-9b-it:free',
                temperature=0.7,
                max_tokens=500
            )
            
            provider = LLMFactory.create(config)
            llm = CodeReviewLLM(provider)
            
            response = llm.generate_review(sample_prompt)
            
            print("\n📝 Review:")
            print(response.content[:300] + "...")
            print(f"\n📊 Stats: {response.total_tokens} tokens, ${response.cost:.4f}, {response.latency:.2f}s")
            
        except Exception as e:
            print(f"❌ OpenRouter test failed: {e}")
    else:
        print("\n⚠️  OPENROUTER_API_KEY not set, skipping OpenRouter test")
    
    # Test Ollama (if running)
    print("\n\n🖥️  Testing Ollama Provider...")
    print("-"*60)
    
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        
        if response.ok:
            config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model='llama3.1:8b',
                temperature=0.7,
                max_tokens=500
            )
            
            provider = LLMFactory.create(config)
            llm = CodeReviewLLM(provider)
            
            response = llm.generate_review(sample_prompt)
            
            print("\n📝 Review:")
            print(response.content[:300] + "...")
            print(f"\n📊 Stats: {response.total_tokens} tokens, FREE (local), {response.latency:.2f}s")
        else:
            print("⚠️  Ollama not running at localhost:11434")
            
    except requests.exceptions.RequestException:
        print("⚠️  Ollama not accessible. Install from: https://ollama.ai")
        print("   Then run: ollama pull llama3.1:8b")


def main():
    """Main demo."""
    demo_providers()
    
    print("\n\n✨ LLM Integration ready!")
    print("\n💡 Next steps:")
    print("   1. Set OPENROUTER_API_KEY or install Ollama")
    print("   2. Run: python scripts/07_end_to_end_review.py")


if __name__ == '__main__':
    main()