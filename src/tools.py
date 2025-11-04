"""
Tool implementations for Q&A, Summarization, and Extraction
FIXED: Works with Ollama (returns strings not Message objects)
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from src.vector_store import VectorStoreManager
from src.prompts import QA_PROMPT, SUMMARIZE_PROMPT, EXTRACT_PROMPT
import json
import logging

logger = logging.getLogger(__name__)

class BaseTool:
    """Base class for all tools"""
    def __init__(self, llm, vector_store: VectorStoreManager):
        self.llm = llm
        self.vector_store = vector_store

    def retrieve_context(self, query: str, k: int = 3) -> tuple[str, List[Dict]]:
        """Retrieve relevant document chunks"""
        docs = self.vector_store.retrieve_documents(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
        return context, sources

    def _extract_response(self, response):
        """Extract text from response (works with both Ollama strings and OpenAI Messages)"""
        if isinstance(response, str):
            return response
        elif hasattr(response, "content"):
            return response.content
        else:
            return str(response)


class QATool(BaseTool):
    """Q&A Tool - Answers specific questions"""
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute Q&A query"""
        try:
            # Retrieve context
            context, sources = self.retrieve_context(query)
            logger.info(f"Retrieved {len(sources)} documents for query: {query}...")
            
            # Prepare messages
            messages = [
                SystemMessage(content=QA_PROMPT),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
            ]
            
            # Get response
            response = self.llm.invoke(messages)
            answer = self._extract_response(response)  # ✅ Works with both Ollama and OpenAI
            
            return {
                "response": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"QATool error: {e}")
            raise


class SummarizeTool(BaseTool):
    """Summarize Tool - Generates summaries"""
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute summarization"""
        try:
            # Retrieve context
            context, sources = self.retrieve_context(query)
            logger.info(f"Retrieved {len(sources)} documents for summarization: {query}...")
            
            # Prepare messages
            messages = [
                SystemMessage(content=SUMMARIZE_PROMPT),
                HumanMessage(content=f"Content to summarize:\n{context}\n\nFocus: {query}")
            ]
            
            # Get response
            response = self.llm.invoke(messages)
            summary = self._extract_response(response)  # ✅ Works with both Ollama and OpenAI
            
            return {
                "response": summary,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"SummarizeTool error: {e}")
            raise


class ExtractTool(BaseTool):
    """Extract Tool - Extracts structured data"""
    
    def execute(self, query: str, output_schema: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Execute extraction"""
        try:
            # Retrieve context
            context, sources = self.retrieve_context(query)
            logger.info(f"Retrieved {len(sources)} documents for extraction: {query}...")
            
            # Build extraction prompt
            schema_prompt = f"\nOutput schema: {json.dumps(output_schema)}" if output_schema else ""
            extraction_prompt = f"{EXTRACT_PROMPT}{schema_prompt}"
            
            # Prepare messages
            messages = [
                SystemMessage(content=extraction_prompt),
                HumanMessage(content=f"Data to extract from:\n{context}\n\nExtraction task: {query}")
            ]
            
            # Get response
            response = self.llm.invoke(messages)
            response_text = self._extract_response(response)  # ✅ Works with both Ollama and OpenAI
            
            # Try to extract JSON
            extracted_data = None
            try:
                # Look for JSON in the response
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    extracted_data = json.loads(json_str)
                elif "{" in response_text and "}" in response_text:
                    # Try to extract JSON from response
                    start = response_text.find("{")
                    end = response_text.rfind("}") + 1
                    if start >= 0 and end > start:
                        json_str = response_text[start:end]
                        extracted_data = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON from response, returning raw text")
                extracted_data = None
            
            return {
                "response": response_text,
                "extracted_data": extracted_data,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"ExtractTool error: {e}")
            raise