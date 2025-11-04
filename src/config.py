"""Configuration - Ollama Local LLM (NO OpenAI Key Needed)"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings with Ollama support"""
    
    # LLM Configuration - Ollama (FREE LOCAL)
    use_local_llm: bool = True
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "phi:latest"
    
    # Embedding Configuration - Ollama (FREE LOCAL)
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    
    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    top_k_results: int = 3
    
    # Vector Store
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "innovate_inc_documents"
    
    # Application
    app_name: str = "AI Market Analyst Agent"
    temperature: float = 0.0
    debug: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # ✅ ALLOW extra environment variables!

settings = Settings()