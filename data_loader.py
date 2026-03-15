"""
SuperGen Universal Data Loader
================================
Import data eksternal ke ChromaDB (vector) dan Neo4j (graph).

Mendukung format JSON dari proyek HybridKnowledge dan AutoGen.

PENGGUNAAN:
  python data_loader.py <file.json> [--target travel|hr] [--clear]

CONTOH:
  python data_loader.py ./HybridKnowledge/vietnam_travel_dataset.json --target travel
  python data_loader.py ./data/candidates.json --target hr
  python data_loader.py ./data/japan_travel.json --target travel --clear

FORMAT JSON YANG DIDUKUNG:

  1) HybridKnowledge Format (Array of objects):
     [
       {
         "id": "city_hanoi",
         "type": "City",
         "name": "Hanoi",
         "description": "...",
         "region": "Northern Vietnam",
         "tags": ["culture", "food"],
         "semantic_text": "...",
         "connections": [
           {"relation": "Connected_To", "target": "city_hue"}
         ]
       }
     ]

  2) Simple Travel Format (from seed_data.py):
     [
       {
         "id": "loc_bali",
         "name": "Bali",
         "type": "Island",
         "country": "Indonesia",
         "description": "...",
         "best_time": "April-Oktober",
         "highlights": ["Tanah Lot", "Ubud"]
       }
     ]

  3) HR/Candidate Format:
     [
       {
         "id": "candidate_001",
         "name": "Budi Santoso",
         "position": "Python Developer",
         "skills": ["Python", "FastAPI"],
         "experience_years": 5,
         "education": "...",
         "summary": "...",
         "location": "Jakarta"
       }
     ]
"""
import sys
import os
import json
import argparse
import logging
import requests
from datetime import datetime, timezone

logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db_unified")
HR_COLLECTION = os.getenv("CHROMA_HR_COLLECTION", "resumes")
TRAVEL_COLLECTION = os.getenv("CHROMA_TRAVEL_COLLECTION", "travel-docs")


# ─────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list:
    """Get embedding via Ollama API."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_MODEL, "prompt": text},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json().get("embedding", [])
    except Exception as e:
        print(f"    [WARN] Embedding error: {e}")
    return []


# ─────────────────────────────────────────────────────────────
# FORMAT DETECTION
# ─────────────────────────────────────────────────────────────
def detect_format(data: list) -> str:
    """Auto-detect format JSON berdasarkan field yang ada."""
    if not data:
        return "unknown"

    sample = data[0]
    # HR format: has 'position' or 'skills'
    if "position" in sample or "skills" in sample:
        return "hr"
    # HybridKnowledge format: has 'connections' and 'semantic_text'
    if "connections" in sample or "semantic_text" in sample:
        return "hybrid_knowledge"
    # Simple travel format: has 'country' or 'highlights'
    if "country" in sample or "highlights" in sample or "best_time" in sample:
        return "simple_travel"
    # Generic: has 'description'
    if "description" in sample and "name" in sample:
        return "generic"
    return "unknown"


# ─────────────────────────────────────────────────────────────
# DOCUMENT BUILDERS
# ─────────────────────────────────────────────────────────────
def build_travel_doc(item: dict, fmt: str) -> tuple:
    """Build (doc_text, metadata) for travel ChromaDB entry."""

    if fmt == "hybrid_knowledge":
        doc = (
            f"Name: {item.get('name', '')}\n"
            f"Type: {item.get('type', '')}\n"
            f"Region: {item.get('region', '')}\n"
            f"City: {item.get('city', '')}\n"
            f"Tags: {', '.join(item.get('tags', []))}\n"
            f"Best Time: {item.get('best_time_to_visit', '')}\n\n"
            f"Description: {item.get('description', '')}\n\n"
            f"{item.get('semantic_text', '')}"
        )
        meta = {
            "name": str(item.get("name", ""))[:200],
            "type": str(item.get("type", ""))[:50],
            "country": str(item.get("region", item.get("country", "")))[:50],
            "title": str(item.get("name", ""))[:200],
        }
        if item.get("tags"):
            meta["tags"] = ",".join(item["tags"])[:200]
        if item.get("city"):
            meta["city"] = str(item["city"])[:50]
    else:  # simple_travel or generic
        highlights = item.get("highlights", [])
        doc = (
            f"Destinasi: {item.get('name', '')}\n"
            f"Tipe: {item.get('type', '')}\n"
            f"Negara: {item.get('country', '')}\n"
            f"Waktu Terbaik: {item.get('best_time', '')}\n"
            f"Highlight: {', '.join(highlights) if isinstance(highlights, list) else str(highlights)}\n\n"
            f"Deskripsi: {item.get('description', '')}"
        )
        meta = {
            "name": str(item.get("name", ""))[:200],
            "type": str(item.get("type", ""))[:50],
            "country": str(item.get("country", ""))[:50],
            "title": str(item.get("name", ""))[:200],
        }

    return doc, meta


def build_hr_doc(item: dict) -> tuple:
    """Build (doc_text, metadata) for HR ChromaDB entry."""
    skills = item.get("skills", [])
    doc = (
        f"Nama: {item.get('name', '')}\n"
        f"Posisi: {item.get('position', '')}\n"
        f"Skills: {', '.join(skills) if isinstance(skills, list) else str(skills)}\n"
        f"Pengalaman: {item.get('experience_years', '')} tahun\n"
        f"Pendidikan: {item.get('education', '')}\n"
        f"Lokasi: {item.get('location', '')}\n"
        f"Tersedia: {'Ya' if item.get('available', True) else 'Tidak'}\n\n"
        f"Ringkasan: {item.get('summary', item.get('description', ''))}"
    )
    meta = {
        "name": str(item.get("name", ""))[:200],
        "position": str(item.get("position", ""))[:100],
        "skills": str(", ".join(skills) if isinstance(skills, list) else str(skills))[:200],
        "location": str(item.get("location", ""))[:100],
    }
    return doc, meta


def enrich_metadata(meta: dict, version: str = "1", source_type: str = "database") -> dict:
    """Add temporal metadata for Conflict Resolution support."""
    meta["uploaded_at"] = datetime.now(timezone.utc).isoformat()
    meta["version"] = str(version)
    meta["source_type"] = str(source_type)
    return meta


# ─────────────────────────────────────────────────────────────
# CHROMADB LOADER
# ─────────────────────────────────────────────────────────────
def load_to_chroma(data: list, target: str, fmt: str, clear: bool = False):
    """Load data array into ChromaDB."""
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col_name = TRAVEL_COLLECTION if target == "travel" else HR_COLLECTION
    collection = client.get_or_create_collection(name=col_name, metadata={"hnsw:space": "cosine"})

    if clear:
        # Delete and recreate
        client.delete_collection(col_name)
        collection = client.get_or_create_collection(name=col_name, metadata={"hnsw:space": "cosine"})
        print(f"  [CLEARED] Collection '{col_name}' direset.")

    print(f"\n[ChromaDB] Loading {len(data)} items -> collection '{col_name}'...")

    success, failed = 0, 0
    for i, item in enumerate(data):
        item_id = item.get("id", f"item_{i}")
        item_name = item.get("name", item_id)

        # Build document text + metadata
        if target == "hr":
            doc, meta = build_hr_doc(item)
        else:
            doc, meta = build_travel_doc(item, fmt)

        # Enrich with temporal metadata for Conflict Resolution
        meta = enrich_metadata(meta, version=args_version, source_type=args_source_type)

        # Generate embedding via Ollama
        emb = get_embedding(doc)

        try:
            if emb:
                try:
                    collection.add(ids=[item_id], documents=[doc], embeddings=[emb], metadatas=[meta])
                except Exception:
                    collection.update(ids=[item_id], documents=[doc], embeddings=[emb], metadatas=[meta])
            else:
                try:
                    collection.add(ids=[item_id], documents=[doc], metadatas=[meta])
                except Exception:
                    collection.update(ids=[item_id], documents=[doc], metadatas=[meta])
            success += 1
        except Exception as e:
            print(f"    [FAIL] {item_name}: {e}")
            failed += 1

        # Progress
        if (i + 1) % 25 == 0 or (i + 1) == len(data):
            print(f"  Progress: {i+1}/{len(data)} (ok={success}, fail={failed})")

    print(f"\n[ChromaDB] Done! {success} loaded, {failed} failed. Total in collection: {collection.count()}")


# ─────────────────────────────────────────────────────────────
# NEO4J LOADER
# ─────────────────────────────────────────────────────────────
def load_to_neo4j(data: list, target: str, fmt: str, clear: bool = False):
    """Load data array into Neo4j knowledge graph."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except Exception as e:
        print(f"\n[Neo4j] SKIP - cannot connect: {e}")
        return

    with driver.session() as session:
        if clear:
            if target == "travel":
                session.run("MATCH (n:Location) DETACH DELETE n")
            else:
                session.run("MATCH (n:Candidate) DETACH DELETE n")
                session.run("MATCH (n:Skill) DETACH DELETE n")
            print(f"  [CLEARED] Old {target} data removed from Neo4j.")

        if target == "hr":
            _load_hr_to_neo4j(session, data)
        else:
            _load_travel_to_neo4j(session, data, fmt)

        # Stats
        result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt ORDER BY cnt DESC")
        print("\n[Neo4j] Node summary:")
        for row in result:
            print(f"  {row['label']}: {row['cnt']}")

    driver.close()


def _load_travel_to_neo4j(session, data, fmt):
    """Load travel locations to Neo4j, supporting both HybridKnowledge and simple formats."""
    print(f"\n[Neo4j] Loading {len(data)} travel entries...")

    nodes_created = 0
    rels_created = 0

    for item in data:
        item_id = item.get("id", item.get("name", "unknown"))
        item_type = item.get("type", "Location")
        name = item.get("name", "Unknown")

        # Create node - support multiple label types (City, Attraction, Hotel, Activity, Location)
        label = item_type if item_type in ("City", "Attraction", "Hotel", "Activity", "Restaurant") else "Location"
        session.run(f"""
            MERGE (n:{label} {{id: $id}})
            SET n.name = $name,
                n.type = $type,
                n.description = $desc,
                n.region = $region,
                n.country = $country,
                n.city = $city,
                n.best_time = $best_time,
                n.tags = $tags
        """, {
            "id": item_id,
            "name": name,
            "type": item_type,
            "desc": item.get("description", ""),
            "region": item.get("region", ""),
            "country": item.get("country", ""),
            "city": item.get("city", ""),
            "best_time": item.get("best_time_to_visit", item.get("best_time", "")),
            "tags": ",".join(item.get("tags", [])) if isinstance(item.get("tags"), list) else str(item.get("tags", "")),
        })
        nodes_created += 1

        # Create connections/relationships
        connections = item.get("connections", [])
        for conn in connections:
            rel_type = conn.get("relation", "RELATED_TO").replace(" ", "_").upper()
            target_id = conn.get("target", "")
            if target_id:
                try:
                    session.run(f"""
                        MATCH (a {{id: $src}})
                        MERGE (b:Location {{id: $dst}})
                        MERGE (a)-[:{rel_type}]->(b)
                    """, {"src": item_id, "dst": target_id})
                    rels_created += 1
                except Exception as e:
                    print(f"    [WARN] Relationship {item_id}->{target_id}: {e}")

    # Create full-text index
    try:
        session.run("DROP INDEX locationFullTextIndex IF EXISTS")
        session.run("""
            CREATE FULLTEXT INDEX locationFullTextIndex
            FOR (n:Location) ON EACH [n.name, n.description, n.type, n.country]
        """)
        print("\n  Full-text index created.")
    except Exception:
        pass

    print(f"\n[Neo4j] Travel: {nodes_created} nodes, {rels_created} relationships created.")


def _load_hr_to_neo4j(session, data):
    """Load HR candidates to Neo4j."""
    print(f"\n[Neo4j] Loading {len(data)} HR candidates...")

    for c in data:
        cid = c.get("id", c.get("name", "unknown"))
        session.run("""
            MERGE (c:Candidate {id: $id})
            SET c.name = $name, c.position = $pos,
                c.experience_years = $exp, c.education = $edu,
                c.location = $loc, c.available = $avail,
                c.summary = $summary
        """, {
            "id": cid, "name": c.get("name", ""),
            "pos": c.get("position", ""), "exp": c.get("experience_years", 0),
            "edu": c.get("education", ""), "loc": c.get("location", ""),
            "avail": c.get("available", True),
            "summary": c.get("summary", c.get("description", ""))
        })

        skills = c.get("skills", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",")]
        for skill in skills:
            session.run("""
                MERGE (s:Skill {name: $skill})
                WITH s MATCH (c:Candidate {id: $id}) MERGE (c)-[:HAS_SKILL]->(s)
            """, {"skill": skill, "id": cid})

    print(f"[Neo4j] HR: {len(data)} candidates loaded.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SuperGen Universal Data Loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CONTOH:
  python data_loader.py ./HybridKnowledge/vietnam_travel_dataset.json --target travel
  python data_loader.py ./data/candidates.json --target hr
  python data_loader.py ./my_data.json --target travel --clear
        """
    )
    parser.add_argument("file", help="Path ke file JSON yang berisi array data")
    parser.add_argument("--target", choices=["travel", "hr"], default=None,
                        help="Target domain: 'travel' atau 'hr' (auto-detect jika tidak diisi)")
    parser.add_argument("--clear", action="store_true",
                        help="Hapus data lama sebelum import (HATI-HATI!)")
    parser.add_argument("--skip-chroma", action="store_true",
                        help="Skip loading ke ChromaDB")
    parser.add_argument("--skip-neo4j", action="store_true",
                        help="Skip loading ke Neo4j")
    parser.add_argument("--version", dest="doc_version", default="1",
                        help="Nomor versi dokumen (untuk conflict resolution)")
    parser.add_argument("--source-type", dest="source_type", default="database",
                        help="Tipe sumber: kebijakan_resmi, memo, email, database")
    parser.add_argument("--supersedes", default=None,
                        help="ID dokumen lama yang digantikan oleh import ini")

    args = parser.parse_args()

    # Validate file
    if not os.path.exists(args.file):
        print(f"[ERROR] File tidak ditemukan: {args.file}")
        sys.exit(1)

    print("=" * 60)
    print("  SUPERGEN DATA LOADER")
    print("=" * 60)
    print(f"  File  : {args.file}")
    print(f"  Size  : {os.path.getsize(args.file) / 1024:.1f} KB")

    # Load JSON
    with open(args.file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("[ERROR] File harus berisi JSON array ([...])")
        sys.exit(1)

    print(f"  Items : {len(data)}")

    # Detect format
    fmt = detect_format(data)
    print(f"  Format: {fmt}")

    # Auto-detect target if needed
    target = args.target
    if not target:
        if fmt == "hr":
            target = "hr"
        else:
            target = "travel"
    print(f"  Target: {target}")
    print(f"  Clear : {'Ya' if args.clear else 'Tidak'}")

    # Check Ollama
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        print(f"  Ollama: Connected (model: {OLLAMA_MODEL})")
    except Exception:
        print(f"  Ollama: NOT CONNECTED (embeddings will be skipped)")

    print("=" * 60)

    # Load to ChromaDB
    if not args.skip_chroma:
        load_to_chroma(data, target, fmt, clear=args.clear,
                       args_version=args.doc_version, args_source_type=args.source_type)
    else:
        print("\n[ChromaDB] Skipped.")

    # Load to Neo4j
    if not args.skip_neo4j:
        load_to_neo4j(data, target, fmt, clear=args.clear)
    else:
        print("\n[Neo4j] Skipped.")

    # Handle supersedes relationship
    if args.supersedes:
        try:
            from core.graph_db import graph_db
            graph_db.connect()
            # Register both documents and create SUPERSEDES relationship
            new_doc_id = f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            graph_db.register_document(
                doc_id=new_doc_id,
                name=os.path.basename(args.file),
                uploaded_at=datetime.now(timezone.utc).isoformat(),
                version=args.doc_version,
                source_type=args.source_type,
                domain=target
            )
            graph_db.mark_superseded(new_doc_id, args.supersedes,
                                     reason=f"Replaced by {os.path.basename(args.file)}")
            print(f"\n[Conflict Resolution] Document '{new_doc_id}' supersedes '{args.supersedes}'")
        except Exception as e:
            print(f"\n[WARN] Could not register supersedes: {e}")

    print("\n" + "=" * 60)
    print("  LOADING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
