import os
from dotenv import load_dotenv

# Load main .env file from the root directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

class Config:
    #LLM Config (Ollama) 
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    # API endpoints
    OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/v1" # OpenAI compatible endpoint

    #Vector DB Config (ChromaDB)
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db_unified")
    
    # Collections
    CHROMA_HR_COLLECTION = os.getenv("CHROMA_HR_COLLECTION", "resumes")
    CHROMA_TRAVEL_COLLECTION = os.getenv("CHROMA_TRAVEL_COLLECTION", "travel-docs")

    # Local Embeddings
    EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

    # --- Knowledge Graph Config (Neo4j) ---
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
    
    @classmethod
    def validate(cls):
        required = ["NEO4J_PASSWORD"]
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing required config keys: {', '.join(missing)}")

config = Config()
