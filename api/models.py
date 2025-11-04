from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any

class QueryRequest(BaseModel):
    """Request model for queries"""
    query: str = Field(..., description="User query")
    use_autonomous_routing: bool = Field(
        default=True,
        description="Use autonomous routing"
    )
    output_schema: Optional[Dict] = Field(default=None, description="JSON schema for extraction")

class SourceDocument(BaseModel):
    """Source document model"""
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    """Response model for queries"""
    query: str
    tool_used: Optional[str] = None
    response: Optional[str] = None
    extracted_data: Optional[Dict] = None
    sources: List[SourceDocument] = []
    processing_time: float
    autonomous_routing: bool = True
    error: Optional[str] = None
