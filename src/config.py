"""
Configuration settings for the AI Market Analyst Agent
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    OPENAI_API_KEY: str
    
    # Model Configuration
    LLM_MODEL: str = "gpt-3.5-turbo"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    TEMPERATURE: float = 0.0
    
    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Retrieval Configuration
    TOP_K_RESULTS: int = 3
    
    # Vector Store
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "innovate_inc_documents"
    
    # Application
    APP_NAME: str = "AI Market Analyst Agent"
    DEBUG: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
