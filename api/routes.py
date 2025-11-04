from fastapi import APIRouter, HTTPException
from api.models import QueryRequest, QueryResponse
from src.agent import MarketAnalystAgent
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Global agent instance (initialized in main.py)
agent: MarketAnalystAgent = None


def get_agent():
    """Get the global agent instance"""
    if agent is None:
        raise HTTPException(
            status_code=500,
            detail="Agent not initialized"
        )
    return agent


@router.post("/query", response_model=QueryResponse)
async def autonomous_query(request: QueryRequest):
    """
    Process query with autonomous routing
    The agent automatically selects the appropriate tool based on
    the query intent.
    """
    try:
        agent_instance = get_agent()
        result = agent_instance.process_query(
            query=request.query,
            use_autonomous_routing=request.use_autonomous_routing
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error in autonomous query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qa")
async def qa_endpoint(request: QueryRequest):
    """Q&A tool endpoint - Answer specific questions"""
    try:
        agent_instance = get_agent()
        result = agent_instance.process_query(
            query=request.query,
            explicit_tool="qa_tool"
        )
        return result
    except Exception as e:
        logger.error(f"Error in QA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize")
async def summarize_endpoint(request: QueryRequest):
    """
    Summarization tool endpoint - Generate summaries
    """
    try:
        agent_instance = get_agent()
        result = agent_instance.process_query(
            query=request.query,
            explicit_tool="summarize_tool"
        )
        return result
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract")
async def extract_endpoint(request: QueryRequest):
    """
    Extraction tool endpoint - Extract structured data
    """
    try:
        agent_instance = get_agent()
        result = agent_instance.process_query(
            query=request.query,
            explicit_tool="extract_tool"
        )
        return result
    except Exception as e:
        logger.error(f"Error in extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
