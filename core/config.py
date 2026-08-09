import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings managed via Pydantic BaseSettings.
    Automatically reads environment variables from .env file or environment.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # General App Config
    APP_NAME: str = Field(default="Faculty Assistant RAG")
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=True)
    LOG_LEVEL: str = Field(default="INFO")

    # API Configuration
    API_V1_PREFIX: str = Field(default="/api/v1")
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])

    # Groq API Configuration
    GROQ_API_KEY: str = Field(..., description="API key for Groq LLM service")
    GROQ_MODEL_NAME: str = Field(default="llama3-70b-8192", description="Default Groq LLM model")

    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model name or path for embeddings"
    )
    EMBEDDING_DIMENSION: int = Field(default=384, description="Vector dimension of embeddings")

    # Pinecone Vector DB Configuration
    PINECONE_API_KEY: str = Field(..., description="Pinecone API key")
    PINECONE_ENVIRONMENT: Optional[str] = Field(default=None, description="Pinecone environment/region")
    PINECONE_INDEX_NAME: str = Field(default="faculty-assistant-index", description="Target Pinecone index")

    # RAG & Generation Parameters
    TOP_K_RETRIEVAL: int = Field(default=5, description="Number of context chunks to retrieve")
    MAX_TOKENS: int = Field(default=1024, description="Maximum completion tokens for LLM generation")
    TEMPERATURE: float = Field(default=0.1, description="LLM sampling temperature")

    # LangSmith / Observability
    LANGCHAIN_TRACING_V2: bool = Field(default=False)
    LANGCHAIN_ENDPOINT: str = Field(default="https://api.smith.langchain.com")
    LANGCHAIN_API_KEY: Optional[str] = Field(default=None)
    LANGCHAIN_PROJECT: str = Field(default="faculty-assistant")


# Global Settings Instance
settings = Settings()