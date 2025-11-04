"""AI Market Analyst Agent - With Improved Routing"""
from typing import Dict, Any, List, Optional
from langchain_community.llms import Ollama
from src.config import settings
from src.vector_store import VectorStoreManager
from src.tools import QATool, SummarizeTool, ExtractTool
import logging
import time

logger = logging.getLogger(__name__)

def get_llm():
    """Get Ollama LLM"""
    logger.info(f"Using Ollama LLM: {settings.ollama_model}")
    return Ollama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.temperature,
        top_p=0.9
    )

class MarketAnalystAgent:
    """Main agent with improved routing"""
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.llm = get_llm()
        
        self.qa_tool = QATool(self.llm, self.vector_store)
        self.summarize_tool = SummarizeTool(self.llm, self.vector_store)
        self.extract_tool = ExtractTool(self.llm, self.vector_store)
        
        self.tools = {
            "qa_tool": self.qa_tool,
            "summarize_tool": self.summarize_tool,
            "extract_tool": self.extract_tool
        }
        
    def route_query(self, query: str) -> tuple[str, dict]:
        """
        Route query to appropriate tool using keyword matching
        Works reliably with smaller models like Phi
        """
        logger.info(f"Routing query: {query}")
        
        query_lower = query.lower()
        
        # EXTRACT keywords - highest priority
        extract_keywords = [
            "extract", "list all", "get all", "find all", "all ", 
            "json", "table", "data", "compile", "gather",
            "market share", "competitor", "competitor", "comparison",
            "statistics", "percentage", "numbers"
        ]
        
        # SUMMARIZE keywords
        summarize_keywords = [
            "summarize", "summary", "overview", "explain", "describe",
            "brief", "concise", "what is", "tell me about",
            "understand", "how would you", "general"
        ]
        
        # Check extract first (most specific)
        if any(keyword in query_lower for keyword in extract_keywords):
            logger.info(f"✅ Routing to extract_tool (matched keywords)")
            return "extract_tool", {"query": query}
        
        # Check summarize
        if any(keyword in query_lower for keyword in summarize_keywords):
            logger.info(f"✅ Routing to summarize_tool (matched keywords)")
            return "summarize_tool", {"query": query}
        
        # Default to QA
        logger.info(f"✅ Routing to qa_tool (default)")
        return "qa_tool", {"query": query}
    
    def execute_tool(self, tool_name: str, arguments: dict) -> Dict[str, Any]:
        """Execute the selected tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        tool = self.tools[tool_name]
        return tool.execute(**arguments)
    
    def process_query(
        self, 
        query: str, 
        use_autonomous_routing: bool = True,
        explicit_tool: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main entry point for processing queries"""
        start_time = time.time()
        
        try:
            # Determine which tool to use
            if explicit_tool:
                tool_name = explicit_tool
                arguments = {"query": query}
            elif use_autonomous_routing:
                tool_name, arguments = self.route_query(query)
            else:
                tool_name = "qa_tool"
                arguments = {"query": query}
            
            # Execute the tool
            result = self.execute_tool(tool_name, arguments)
            processing_time = time.time() - start_time
            
            response = {
                "query": query,
                "tool_used": tool_name,
                "response": result.get("response"),
                "extracted_data": result.get("extracted_data"),
                "sources": result.get("sources", []),
                "processing_time": round(processing_time, 3),
                "autonomous_routing": use_autonomous_routing
            }
            
            logger.info(f"Query processed in {processing_time:.3f}s using {tool_name}")
            return response
            
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "processing_time": time.time() - start_time
            }


def create_agent(document_text: str = None) -> MarketAnalystAgent:
    """Factory function to create agent"""
    from src.vector_store import initialize_document_store
    
    if document_text:
        vector_store = initialize_document_store(document_text)
    else:
        vector_store = VectorStoreManager()
        vector_store.load_vector_store()
    
    agent = MarketAnalystAgent(vector_store)
    logger.info("Agent created successfully with improved routing")
    return agent