"""
Tool implementations for Q&A, Summarization, and Extraction
"""
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from src.vector_store import VectorStoreManager
from src.prompts import QA_PROMPT, SUMMARIZE_PROMPT, EXTRACT_PROMPT
import json
import logging

logger = logging.getLogger(__name__)


class BaseTool:
    """Base class for all tools"""
    
    def __init__(self, llm: ChatOpenAI, vector_store: VectorStoreManager):
        self.llm = llm
        self.vector_store = vector_store
        
    def retrieve_context(self, query: str, k: int = 3) -> tuple[str, List[Dict]]:
        """
        Retrieve relevant document chunks
        
        Returns:
            Tuple of (combined_context, source_documents)
        """
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


class QATool(BaseTool):
    """Tool for answering specific questions"""
    
    def execute(self, query: str) -> Dict[str, Any]:
        """
        Answer a specific question about the document
        
        Args:
            query: Question to answer
            
        Returns:
            Dictionary with response and sources
        """
        logger.info(f"QA Tool executing: {query}")
        
        # Retrieve relevant context
        context, sources = self.retrieve_context(query)
        
        # Generate answer
        prompt = QA_PROMPT.format(context=context, question=query)
        messages = [HumanMessage(content=prompt)]
        
        response = self.llm.predict_messages(messages)
        answer = response.content
        
        return {
            "response": answer,
            "sources": sources,
            "tool": "qa_tool"
        }


class SummarizeTool(BaseTool):
    """Tool for generating summaries"""
    
    def execute(self, query: str) -> Dict[str, Any]:
        """
        Generate a comprehensive summary
        
        Args:
            query: Topic or section to summarize
            
        Returns:
            Dictionary with summary and sources
        """
        logger.info(f"Summarize Tool executing: {query}")
        
        # Retrieve relevant context (get more chunks for summaries)
        context, sources = self.retrieve_context(query, k=5)
        
        # Generate summary
        prompt = SUMMARIZE_PROMPT.format(context=context, topic=query)
        messages = [HumanMessage(content=prompt)]
        
        response = self.llm.predict_messages(messages)
        summary = response.content
        
        return {
            "response": summary,
            "sources": sources,
            "tool": "summarize_tool"
        }


class ExtractTool(BaseTool):
    """Tool for extracting structured data"""
    
    def execute(
        self, 
        query: str, 
        schema: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data in JSON format
        
        Design Decision - Structured JSON Extraction:
        
        Uses a multi-step approach to ensure reliable JSON output:
        1. Retrieve relevant document chunks
        2. Provide explicit schema instructions
        3. Use JSON mode or careful prompting
        4. Validate output format
        5. Return with confidence scores
        
        Args:
            query: What data to extract
            schema: Optional JSON schema for output structure
            
        Returns:
            Dictionary with extracted data and sources
        """
        logger.info(f"Extract Tool executing: {query}")
        
        # Retrieve relevant context
        context, sources = self.retrieve_context(query, k=4)
        
        # Prepare schema instruction
        schema_instruction = ""
        if schema:
            schema_instruction = f"\n\nRequired output schema:\n{json.dumps(schema, indent=2)}"
        
        # Generate extraction prompt
        prompt = EXTRACT_PROMPT.format(
            context=context,
            query=query,
            schema=schema_instruction
        )
        
        messages = [HumanMessage(content=prompt)]
        
        # Request JSON mode for reliable output
        response = self.llm.predict_messages(
            messages,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        try:
            # Parse JSON response
            extracted_data = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
            else:
                extracted_data = {"raw_response": response.content}
        
        return {
            "extracted_data": extracted_data,
            "sources": sources,
            "tool": "extract_tool"
        }
