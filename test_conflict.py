import os
import json
import logging
import time
from datetime import datetime, timezone, timedelta

# Mock env vars to ensure we don't mess up main DB if running standalone
os.environ["CHROMA_HR_COLLECTION"] = "test_conflict_collection"

from core.vector_db import vector_db
from core.graph_db import graph_db
from core.conflict_resolver import conflict_resolver
import chromadb

def setup_test_data():
    """Setup conflicting data in ChromaDB and Neo4j for testing."""
    print("Setting up test conflict scenario...")
    
    # 1. Initialize
    vector_db.initialize()
    graph_db.connect()
    
    # Clear old test data
    try:
        vector_db.client.delete_collection("test_conflict_collection")
    except Exception:
        pass
    
    test_collection = vector_db.client.get_or_create_collection(
        name="test_conflict_collection", 
        metadata={"hnsw:space": "cosine"}
    )
    vector_db.hr_collection = test_collection  # Hijack for testing
    
    # Clear Neo4j test nodes
    graph_db.query("MATCH (n:Document {domain: 'test_domain'}) DETACH DELETE n")
    
    old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    new_time = datetime.now(timezone.utc).isoformat()
    
    # 2. Insert Document A (Lama)
    doc_a_id = "doc_wre_kebijakan_lama"
    doc_a_text = (
        "Nama: Kebijakan Gaji WRE\n"
        "Posisi: Instruktur WRE\n"
        "Ringkasan: Jadwal Gaji Instruktur WRE adalah tanggal 25 setiap bulan."
    )
    doc_a_meta = {
        "name": "Kebijakan Lama WRE",
        "uploaded_at": old_time,
        "version": "1",
        "source_type": "kebijakan_resmi",
        "title": "Kebijakan Lama WRE"
    }
    emb_a = vector_db._get_embedding(doc_a_text)
    
    # Insert ke ChromaDB
    test_collection.add(
        ids=[doc_a_id], documents=[doc_a_text], 
        metadatas=[doc_a_meta], embeddings=[emb_a]
    )
    
    # Register ke Neo4j
    graph_db.register_document(
        doc_id=doc_a_id, name="Kebijakan Lama WRE", 
        uploaded_at=old_time, version="1", 
        source_type="kebijakan_resmi", domain="test_domain"
    )
    
    # 3. Tunggu sebentar untuk memastikan indexing
    time.sleep(1)
    
    # 4. Insert Document B (Baru)
    doc_b_id = "doc_wre_kebijakan_baru"
    doc_b_text = (
        "Nama: Kebijakan Gaji WRE 2026\n"
        "Posisi: Instruktur WRE\n"
        "Ringkasan: Mulai tahun 2026, Jadwal Gaji Instruktur WRE diubah menjadi tanggal 1."
    )
    doc_b_meta = {
        "name": "Kebijakan Baru WRE 2026",
        "uploaded_at": new_time,
        "version": "2",
        "source_type": "kebijakan_resmi",
        "title": "Kebijakan Baru WRE 2026"
    }
    emb_b = vector_db._get_embedding(doc_b_text)
    
    # Insert ke ChromaDB
    test_collection.add(
        ids=[doc_b_id], documents=[doc_b_text], 
        metadatas=[doc_b_meta], embeddings=[emb_b]
    )
    
    # Register ke Neo4j
    graph_db.register_document(
        doc_id=doc_b_id, name="Kebijakan Baru WRE 2026", 
        uploaded_at=new_time, version="2", 
        source_type="kebijakan_resmi", domain="test_domain"
    )
    
    # 5. BUAT RELASI SUPERSEDES!
    print("Creating SUPERSEDES relationship in Neo4j...")
    graph_db.mark_superseded(doc_b_id, doc_a_id, "Kebijakan gaji diupdate untuk 2026")
    
    return doc_a_id, doc_b_id

def run_test():
    """Run the conflict resolution test pipeline."""
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    
    setup_test_data()
    
    query = "Kapan gaji instruktur WRE dibayarkan?"
    print(f"\n--- Menguji Query: '{query}' ---")
    
    # 1. Raw Vector Search
    print("\n1. Hasil Raw Search ChromaDB (Tanpa Conflict Resolution):")
    raw_results = vector_db.hr_search(query, top_k=2)
    for i, r in enumerate(raw_results):
        print(f"   {i+1}. ID: {r['id']} | Sim Score: {r['score']:.4f} | Teks: {r['text'][:60]}...")
    
    # 2. Conflict Resolution Pipeline
    print("\n2. Memproses dengan Conflict Resolver...")
    resolution = conflict_resolver.resolve(raw_results, graph_db)
    
    print("\n=== HASIL CONFLICT RESOLUTION ===")
    print(f"Log: {resolution['resolution_log']}")
    print(f"Konflik Terdeteksi: {resolution['conflicts_found']}")
    
    final_results = resolution["results"]
    for i, r in enumerate(final_results):
        status = "[USANG]" if r.get("_superseded") else "[AKTIF]"
        print(f"   {i+1}. {status} ID: {r['id']} | Final Score: {r.get('final_score', 0):.4f}")
        print(f"      Breakdown: {r.get('score_breakdown', {})}")
        print(f"      Teks: {r['text'][:80]}...")
        
    print("\nTest completed successfully!")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('.env')
    run_test()
