from core.vector_db import vector_db
from core.graph_db import graph_db
from core.llm_client import llm
from core.conflict_resolver import conflict_resolver
import logging

logger = logging.getLogger(__name__)

class HRAgent:
    """Agent for Recruitment and Candidate analysis — with Conflict Resolution"""
    
    def __init__(self):
        self.name = "AI HR Assistant"
        
    def answer_query(self, query: str):
        sources = []
        chroma_context = ""
        neo4j_context = ""
        conflict_info = ""

        # 1. Retrieve from Vector DB (now returns enriched results)
        vectors = vector_db.hr_search(query, top_k=5)
        if vectors:
            # 2. Run Conflict Resolution on vector results
            resolution = conflict_resolver.resolve(vectors, graph_db)
            resolved_results = resolution["results"]
            
            if resolution["conflicts_found"] > 0:
                conflict_info = f"\n⚡ CONFLICT RESOLUTION: {resolution['resolution_log']}\n"
                logger.info(f"HR Conflict resolved: {resolution['resolution_log']}")
            
            # Build context from resolved (non-superseded) results
            active_results = [r for r in resolved_results if not r.get("_superseded")]
            if active_results:
                chroma_context = "INFO DARI DATABASE RESUME (CHROMA):\n"
                for r in active_results[:3]:
                    chroma_context += r.get("text", "") + "\n...\n"
                sources.append("Vector DB")

        # 3. Retrieve from Graph DB
        graph_results = graph_db.hr_search_candidates(query)
        if graph_results:
            neo4j_context = "INFO DARI KNOWLEDGE GRAPH:\n"
            for record in graph_results:
                neo4j_context += f"- [{record['type']}] {record['entity']}\n"
                for rel in record['relationships']:
                    if rel['rel'] and rel['target']:
                        neo4j_context += f"  -> {rel['rel']}: {rel['target']}\n"
            sources.append("Knowledge Graph")

        # Combine
        full_context = ""
        if chroma_context or neo4j_context:
            full_context = f"{chroma_context}\n\n{neo4j_context}"
            
        system_prompt = f"""Kamu adalah AI HR Assistant profesional. Tugasmu adalah menjawab pertanyaan user terkait rekrutmen atau data kandidat.
Gunakan informasi konteks yang disediakan di bawah ini untuk menjawab pertanyaan. Jika informasinya tidak ada di konteks, katakan bahwa data tidak ditemukan.
{conflict_info}
KONTEKS DATA KANDIDAT:
{full_context if full_context else "Tidak ada data relevan di database saat ini."}
    
Aturan menjawab:
1. Gunakan Bahasa Indonesia yang profesional dan mudah dibaca.
2. Gunakan format markdown (bold, list) jika diperlukan.
3. JANGAN menyebutkan hal teknis seperti "Berdasarkan ChromaDB" atau "Menurut Neo4j", cukup sebutkan datanya.
4. Jika ada catatan CONFLICT RESOLUTION, prioritaskan informasi yang lebih baru."""

        answer = llm.generate(query, system_prompt=system_prompt)
        
        return {
            "answer": answer,
            "sources": sources,
            "conflicts_resolved": conflict_info if conflict_info else None
        }

hr_agent = HRAgent()
