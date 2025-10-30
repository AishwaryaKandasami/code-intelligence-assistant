from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # GitHub Configuration
    github_token: str = Field(..., env='GITHUB_TOKEN')
    github_api_base_url: str = Field(
        default='https://api.github.com',
        env='GITHUB_API_BASE_URL'
    )
    target_repos: str = Field(
        default='tiangolo/fastapi,psf/requests,pydantic/pydantic',
        env='TARGET_REPOS'
    )

    # Data Collection Settings
    max_prs_per_repo: int = Field(default=100, env='MAX_PRS_PER_REPO')
    min_review_comments: int = Field(default=3, env='MIN_REVIEW_COMMENTS')
    days_lookback: int = Field(default=365, env='DAYS_LOOKBACK')

    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, env='OPENAI_API_KEY')
    openai_model: str = Field(default='gpt-4o-mini', env='OPENAI_MODEL')

    anthropic_api_key: Optional[str] = Field(default=None, env='ANTHROPIC_API_KEY')
    anthropic_model: str = Field(
        default='claude-3-haiku-20240307',
        env='ANTHROPIC_MODEL'
    )

    openrouter_api_key: Optional[str] = Field(default=None, env='OPENROUTER_API_KEY')
    openrouter_model: str = Field(
        default='mistralai/mistral-7b-instruct',
        env='OPENROUTER_MODEL'
    )

    use_local_ollama: bool = Field(default=False, env='USE_LOCAL_OLLAMA')
    ollama_base_url: str = Field(default='http://localhost:11434', env='OLLAMA_BASE_URL')
    ollama_model: str = Field(default='codellama', env='OLLAMA_MODEL')

    # Vector Database Configuration
    vector_db_type: str = Field(default='chromadb', env='VECTOR_DB_TYPE')
    chromadb_path: str = Field(
        default='./data/embeddings/chromadb',
        env='CHROMADB_PATH'
    )
    embedding_model: str = Field(
        default='sentence-transformers/all-MiniLM-L6-v2',
        env='EMBEDDING_MODEL'
    )

    # Application Settings
    log_level: str = Field(default='INFO', env='LOG_LEVEL')
    data_dir: str = Field(default='./data', env='DATA_DIR')
    cache_dir: str = Field(default='./cache', env='CACHE_DIR')

    # Streamlit Configuration
    streamlit_server_port: int = Field(default=8501, env='STREAMLIT_SERVER_PORT')
    streamlit_server_address: str = Field(
        default='localhost',
        env='STREAMLIT_SERVER_ADDRESS'
    )

    # Feature Flags
    enable_caching: bool = Field(default=True, env='ENABLE_CACHING')
    enable_metrics: bool = Field(default=True, env='ENABLE_METRICS')
    debug_mode: bool = Field(default=False, env='DEBUG_MODE')


    @validator('target_repos')
    def parse_repos(cls, v):
        """Parse comma-separated repo list."""
        return [repo.strip() for repo in v.split(',') if repo.strip()]

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    @property
    def raw_data_path(self) -> Path:
        return self.data_path / 'raw'

    @property
    def processed_data_path(self) -> Path:
        return self.data_path / 'processed'

    @property
    def embeddings_path(self) -> Path:
        return self.data_path / 'embeddings'

    def get_active_llm_config(self) -> dict:
        """Get configuration for the active LLM provider."""
        if self.use_local_ollama:
            return {
                'provider': 'ollama',
                'base_url': self.ollama_base_url,
                'model': self.ollama_model
            }
        elif self.anthropic_api_key:
            return {
                'provider': 'anthropic',
                'api_key': self.anthropic_api_key,
                'model': self.anthropic_model
            }
        elif self.openrouter_api_key:
            return {
                'provider': 'openrouter',
                'api_key': self.openrouter_api_key,
                'model': self.openrouter_model
            }
        elif self.openai_api_key:
            return {
                'provider': 'openai',
                'api_key': self.openai_api_key,
                'model': self.openai_model
            }
        else:
            raise ValueError("No LLM provider configured. Set API keys in .env file")

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.embeddings_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False


# Global settings instance
settings = Settings()
settings.ensure_directories()
