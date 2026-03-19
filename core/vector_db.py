import chromadb
import requests
import logging
from core.config import config

logger = logging.getLogger(__name__)

class ChromaClient:
    def __init__(self):
        self.client = None
        self.hr_collection = None
        self.travel_collection = None

    def initialize(self):
        logger.info("📦 Initializing ChromaDB client...")
        
        try:
            # We use PersistentClient to maintain data purely locally
            self.client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
            
            # Init HR Vector
            self.hr_collection = self.client.get_or_create_collection(
                name=config.CHROMA_HR_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Init Travel Vector
            self.travel_collection = self.client.get_or_create_collection(
                name=config.CHROMA_TRAVEL_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✅ Connected to local ChromaDB and loaded collections")
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to ChromaDB: {e}")

    def _get_embedding(self, text: str) -> list:
        """Get embedding from Ollama. Supports both new and old API."""
        # Try NEW Ollama API first (/api/embed)
        try:
            resp = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/embed",
                json={"model": config.OLLAMA_MODEL, "input": text},
                timeout=120
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if embeddings and len(embeddings) > 0:
                    return embeddings[0]
                return data.get("embedding", [])
        except Exception:
            pass

        # Fallback to OLD Ollama API (/api/embeddings)
        try:
            resp = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/embeddings",
                json={"model": config.OLLAMA_MODEL, "prompt": text},
                timeout=120
            )
            if resp.status_code == 200:
                return resp.json().get("embedding", [])
        except Exception as e:
            logger.error(f"Failed to get embedding from Ollama: {e}")
        return []

    def hr_search(self, query, top_k=3):
        """Search HR collection. Returns enriched results with metadata for conflict resolution."""
        if not self.hr_collection: return []
        try:
            emb = self._get_embedding(query)
            if not emb: return []
            res = self.hr_collection.query(
                query_embeddings=[emb], 
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            results = []
            if res and res.get("ids") and res["ids"][0]:
                for i, doc_id in enumerate(res["ids"][0]):
                    metadata = res["metadatas"][0][i] if res.get("metadatas") else {}
                    distance = res["distances"][0][i] if res.get("distances") else 0
                    doc = res["documents"][0][i] if res.get("documents") else ""
                    
                    results.append({
                        "id": doc_id,
                        "score": round(1 - distance, 4),
                        "text": doc,
                        "metadata": metadata
                    })
            return results
        except Exception as e:
            logger.error(f"HR Chroma query error: {e}")
            return []

    def travel_search(self, query, top_k=5, filters=None):
        if not self.travel_collection: return []
        try:
            emb = self._get_embedding(query)
            if not emb: return []
            
            where_filter = filters
            
            params = {
                "query_embeddings": [emb],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }
            if where_filter: params["where"] = where_filter
            
            res = self.travel_collection.query(**params)
            
            results = []
            if res and res.get("ids") and res["ids"][0]:
                for i, doc_id in enumerate(res["ids"][0]):
                    metadata = res["metadatas"][0][i] if res.get("metadatas") else {}
                    distance = res["distances"][0][i] if res.get("distances") else 0
                    doc = res["documents"][0][i] if res.get("documents") else ""
                    
                    results.append({
                        "id": doc_id,
                        "score": round(1 - distance, 4),
                        "text": doc or metadata.get("chunk_text", ""),
                        "source": metadata.get("source", ""),
                        "title": metadata.get("title", ""),
                        "country": metadata.get("country", "")
                    })
            return results
        except Exception as e:
            logger.error(f"Travel Chroma query error: {e}")
            return []

# Singleton instance
vector_db = ChromaClient()
