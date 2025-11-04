"""
AI Market Analyst Agent - Main Agent Implementation
Implements autonomous routing and tool orchestration
"""
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from src.config import settings
from src.vector_store import VectorStoreManager
from src.tools import QATool, SummarizeTool, ExtractTool
from src.prompts import (
    ROUTING_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    EXTRACT_SYSTEM_PROMPT
)
import json
import logging
import time

logger = logging.getLogger(__name__)

class MarketAnalystAgent:
    """
    Main agent that orchestrates tool selection and execution
    Implements autonomous routing (Bonus 1)
    """
    
    def __init__(self, vector_store: VectorStoreManager):
        """
        Initialize the agent with vector store and tools
        
        Args:
            vector_store: Initialized VectorStoreManager instance
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize tools
        self.qa_tool = QATool(self.llm, self.vector_store)
        self.summarize_tool = SummarizeTool(self.llm, self.vector_store)
        self.extract_tool = ExtractTool(self.llm, self.vector_store)
        
        # Tool registry
        self.tools = {
            "qa_tool": self.qa_tool,
            "summarize_tool": self.summarize_tool,
            "extract_tool": self.extract_tool
        }
        
        # Tool function definitions for OpenAI function calling
        self.tool_functions = [
            {
                "type": "function",
                "function": {
                    "name": "qa_tool",
                    "description": "Answers specific factual questions about the market research document. Use this for queries asking 'what', 'who', 'when', 'how much', or requesting specific facts and figures.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The specific question to answer"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "summarize_tool",
                    "description": "Generates comprehensive summaries and overviews of document sections or topics. Use this for queries asking to 'summarize', 'overview', 'explain', or requesting high-level insights about broad topics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The topic or section to summarize"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_tool",
                    "description": "Extracts structured data in JSON format from the document. Use this for queries asking to 'extract', 'list all', 'get data about', or requesting information in a structured format.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What data to extract"
                            },
                            "schema": {
                                "type": "object",
                                "description": "Optional JSON schema for the output structure"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
    def route_query(self, query: str) -> tuple[str, dict]:
        """
        Autonomously determine which tool to use based on the query
        
        This implements Bonus 1: Autonomous Routing
        
        Design Decision:
        Uses OpenAI's function calling capability to let the LLM decide
        which tool is most appropriate based on the query characteristics.
        
        Args:
            query: User's natural language query
            
        Returns:
            Tuple of (tool_name, arguments_dict)
        """
        logger.info(f"Routing query: {query}")
        
        messages = [
            SystemMessage(content=ROUTING_SYSTEM_PROMPT),
            HumanMessage(content=f"User query: {query}")
        ]
        
        # Use function calling to determine tool
        response = self.llm.predict_messages(
            messages,
            functions=self.tool_functions,
            function_call="auto"
        )
        
        # Extract function call from response
        if response.additional_kwargs.get("function_call"):
            function_call = response.additional_kwargs["function_call"]
            tool_name = function_call["name"]
            arguments = json.loads(function_call["arguments"])
            
            logger.info(f"Routed to {tool_name} with args: {arguments}")
            return tool_name, arguments
        else:
            # Fallback to QA tool if no function call
            logger.warning("No function call detected, defaulting to qa_tool")
            return "qa_tool", {"query": query}
    
    def execute_tool(
        self, 
        tool_name: str, 
        arguments: dict
    ) -> Dict[str, Any]:
        """
        Execute the selected tool with provided arguments
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution results
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        result = tool.execute(**arguments)
        
        return result
    
    def process_query(
        self, 
        query: str, 
        use_autonomous_routing: bool = True,
        explicit_tool: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing queries
        
        Args:
            query: User's query
            use_autonomous_routing: Whether to use autonomous routing
            explicit_tool: Explicitly specify tool (overrides routing)
            
        Returns:
            Complete response with tool used, results, and metadata
        """
        start_time = time.time()
        
        try:
            # Determine which tool to use
            if explicit_tool:
                tool_name = explicit_tool
                arguments = {"query": query}
            elif use_autonomous_routing:
                tool_name, arguments = self.route_query(query)
            else:
                # Default to QA tool
                tool_name = "qa_tool"
                arguments = {"query": query}
            
            # Execute the tool
            result = self.execute_tool(tool_name, arguments)
            
            # Add metadata
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
            
            logger.info(
                f"Query processed successfully in {processing_time:.3f}s "
                f"using {tool_name}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "processing_time": time.time() - start_time
            }


def create_agent(document_text: str = None) -> MarketAnalystAgent:
    """
    Factory function to create a configured agent instance
    
    Args:
        document_text: Optional document text to initialize vector store
        
    Returns:
        Initialized MarketAnalystAgent
    """
    from src.vector_store import initialize_document_store
    
    if document_text:
        vector_store = initialize_document_store(document_text)
    else:
        # Load existing vector store
        vector_store = VectorStoreManager()
        vector_store.load_vector_store()
    
    agent = MarketAnalystAgent(vector_store)
    logger.info("Agent created successfully")
    
    return agent
