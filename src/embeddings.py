"""
Embedding Models Management
"""
from typing import Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import settings
import logging

logger = logging.getLogger(__name__)

def get_embedding_function(model_name: str = None) -> Any:
    """
    Get embedding function based on model name
    
    Design Decision - Embedding Model: OpenAI text-embedding-3-small
    
    Rationale:
    1. High-quality embeddings with 1536 dimensions
    2. Cost-effective ($0.02 per 1M tokens vs $0.13 for ada-002)
    3. Better performance than ada-002 on MTEB benchmarks
    4. Low latency (~50-100ms per request)
    5. Consistent quality across diverse domains
    6. Native integration with OpenAI ecosystem
    7. No local GPU requirements
    
    Alternative Considered: sentence-transformers/all-MiniLM-L6-v2
    - Pros: Free, runs locally, 384 dimensions (faster)
    - Cons: Lower accuracy, requires local compute, 
            less robust for domain-specific queries
    
    Args:
        model_name: Name of embedding model
        
    Returns:
        Embedding function instance
    """
    model_name = model_name or settings.EMBEDDING_MODEL
    
    if model_name.startswith("text-embedding"):
        # OpenAI embeddings
        logger.info(f"Using OpenAI embedding model: {model_name}")
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.OPENAI_API_KEY
        )
    else:
        # HuggingFace embeddings (for comparison/local deployment)
        logger.info(f"Using HuggingFace embedding model: {model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )


class EmbeddingComparator:
    """Compare different embedding models for retrieval quality"""
    
    def __init__(self, models: list):
        """
        Initialize comparator with list of model names
        
        Args:
            models: List of embedding model names to compare
        """
        self.models = models
        self.embedding_functions = {
            model: get_embedding_function(model) 
            for model in models
        }
        
    def compare_embeddings(self, texts: list, query: str) -> dict:
        """
        Compare embedding quality across models
        
        Args:
            texts: List of text chunks
            query: Query to test retrieval
            
        Returns:
            Comparison results dictionary
        """
        results = {}
        
        for model_name, embed_func in self.embedding_functions.items():
            # Embed query and texts
            query_embedding = embed_func.embed_query(query)
            text_embeddings = embed_func.embed_documents(texts)
            
            # Calculate similarities
            from numpy import dot
            from numpy.linalg import norm
            
            similarities = [
                dot(query_embedding, text_emb) / (norm(query_embedding) * norm(text_emb))
                for text_emb in text_embeddings
            ]
            
            results[model_name] = {
                "similarities": similarities,
                "top_match_idx": similarities.index(max(similarities)),
                "top_similarity": max(similarities)
            }
            
        return results
