from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging
import os

# Set up logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import configuration
from core.config import config
# Import databases
from core.graph_db import graph_db
from core.vector_db import vector_db
# Import agents
from agents.hr_agent import hr_agent
from agents.travel_agent import travel_agent

app = FastAPI(
    title="🚀 SuperGen Modular Server",
    description="Unified API combining HR Recruitment (AutoGen) & Travel Assistant (HybridKnowledge) with Multi-Agent support.",
    version="2.0.0"
)

# Startup event to connect databases
@app.on_event("startup")
async def startup_event():
    try:
        config.validate()
        logger.info("✅ Configuration validated.")
    except Exception as e:
        logger.error(f"❌ Config error: {e}")

    logger.info("📦 Initializing Vector DB (Chroma)...")
    vector_db.initialize()

    logger.info("🔗 Connecting to Knowledge Graph (Neo4j)...")
    graph_db.connect()

    logger.info("🚀 SuperGen Modular Server is READY!")


# Request schemas
class QueryModel(BaseModel):
    query: str = Field(..., min_length=1, description="Your question")


class TravelQueryModel(BaseModel):
    query: str = Field(..., min_length=1, description="Your travel question")
    use_autogen: bool = Field(False, description="Force multi-agent mode (AutoGen team)")


@app.get("/")
def root():
    return {
        "status": "online",
        "description": "SuperGen Modular Ecosystem v2.0",
        "agents": {
            "hr": "POST /hr/chat — AI HR Recruitment Assistant",
            "travel_simple": "POST /travel/chat — Travel Assistant (simple mode)",
            "travel_autogen": "POST /travel/chat {use_autogen: true} — Travel with Multi-Agent Team"
        },
        "docs": "GET /docs"
    }


@app.post("/hr/chat")
def hr_chat(request: QueryModel):
    try:
        response = hr_agent.answer_query(request.query)
        return {"query": request.query, "data": response, "agent": "HR Assistant"}
    except Exception as e:
        logger.error(f"HR Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/travel/chat")
async def travel_chat(request: TravelQueryModel):
    try:
        response = await travel_agent.answer_query(request.query, use_autogen=request.use_autogen)
        return {"query": request.query, "data": response, "agent": "Travel Assistant"}
    except Exception as e:
        logger.error(f"Travel Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Unified SuperGen Server (Modular) v2.0")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
