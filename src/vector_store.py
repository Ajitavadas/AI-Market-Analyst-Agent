"""
Vector Store Management with ChromaDB
"""
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.config import settings
from src.embeddings import get_embedding_function
import logging
import os
os.environ['CHROMA_TELEMETRY_ENABLED'] = 'false'  # Disable telemetry


logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store operations for document storage and retrieval"""
    
    def __init__(self, embedding_model: str = None):
        """
        Initialize the Vector Store Manager
        
        Args:
            embedding_model: Name of the embedding model to use
        """
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.embedding_function = get_embedding_function(self.embedding_model)
        self.vector_store = None
        
    def create_chunks(self, text: str) -> List[Document]:
        """
        Split text into chunks using RecursiveCharacterTextSplitter
        
        Design Decision - Chunking Strategy:
        - Chunk Size: 512 tokens
        - Overlap: 50 tokens (10%)
        
        Rationale:
        1. 512 tokens balances context preservation with retrieval precision
        2. Smaller chunks (256) would lose too much context for complex queries
        3. Larger chunks (1024+) reduce retrieval accuracy and increase noise
        4. 10% overlap ensures continuity at chunk boundaries without redundancy
        5. RecursiveCharacterTextSplitter respects sentence boundaries
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of Document objects with chunked content
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        documents = [
            Document(
                page_content=chunk,
                metadata={"chunk_id": i, "source": "innovate_inc_report"}
            )
            for i, chunk in enumerate(chunks)
        ]
        
        logger.info(f"Created {len(documents)} chunks from input text")
        return documents
    
    def initialize_vector_store(self, documents: List[Document]):
        """
        Initialize ChromaDB vector store with documents
        
        Design Decision - Vector Database: ChromaDB
        
        Rationale:
        1. Lightweight and easy to set up for development and production
        2. Excellent Python integration with minimal configuration
        3. Built-in persistence without external database dependencies
        4. Fast similarity search with HNSW indexing
        5. Supports filtering on metadata
        6. Open-source and free for commercial use
        7. Lower latency than cloud-based solutions (Pinecone)
        8. Better suited for single-tenant applications than Weaviate
        
        Args:
            documents: List of Document objects to index
        """
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.COLLECTION_NAME
        )
        
        logger.info(f"Initialized vector store with {len(documents)} documents")
        
    def load_vector_store(self):
        """Load existing vector store from disk"""
        self.vector_store = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=self.embedding_function,
            collection_name=settings.COLLECTION_NAME
        )
        logger.info("Loaded existing vector store")
        
    def retrieve_documents(self, query: str, k: int = None) -> List[Document]:
        """
        Retrieve most relevant documents for a query
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of most relevant documents
        """
        k = k or settings.TOP_K_RESULTS
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call initialize_vector_store() first.")
        
        results = self.vector_store.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        
        return results
    
    def retrieve_with_scores(self, query: str, k: int = None) -> List[tuple]:
        """
        Retrieve documents with similarity scores
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        k = k or settings.TOP_K_RESULTS
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized.")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results


def initialize_document_store(document_text: str, embedding_model: str = None) -> VectorStoreManager:
    """
    Initialize vector store with document text
    
    Args:
        document_text: Text content to index
        embedding_model: Embedding model to use
        
    Returns:
        Initialized VectorStoreManager
    """
    manager = VectorStoreManager(embedding_model=embedding_model)
    documents = manager.create_chunks(document_text)
    manager.initialize_vector_store(documents)
    return manager
