from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from api.routes import set_agent
from src.agent import create_agent
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize FastAPI app
app = FastAPI(
    title="AI Market Analyst Agent API",
    description="Multi-functional AI agent for market research analysis",
    version="1.0.0"
)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routes
app.include_router(router, prefix="/api")
# Initialize agent on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the agent and vector store"""
    try:
        # Load document
        with open("data/innovate_inc_report.txt", "r") as f:
            document_text = f.read()
        # Create agent
        global agent
        agent = create_agent(document_text)
        set_agent(agent)
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Market Analyst Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}