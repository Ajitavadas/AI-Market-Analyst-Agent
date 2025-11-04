"""Embeddings - Ollama Only (No HuggingFace, No OpenAI)"""
from typing import Any
import logging

logger = logging.getLogger(__name__)

def get_embedding_function(model_name: str = None) -> Any:
    """Get Ollama embeddings - FREE, no API key needed"""
    
    from langchain_community.embeddings import OllamaEmbeddings
    from src.config import settings
    
    model_name = model_name or settings.embedding_model
    
    logger.info(f"✅ Using Ollama embedding model: {model_name}")
    return OllamaEmbeddings(
        model=model_name,
        base_url=settings.ollama_base_url
    )