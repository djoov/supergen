"""
SuperGen Data Seeder (Standalone, Ollama Embeddings)
=====================================================
Mengisi data sample ke ChromaDB dan Neo4j.
Menggunakan Ollama API untuk embeddings (tidak bergantung sentence-transformers).

Jalankan:
    python seed_data.py
"""
import sys
import os
import json
import logging
import requests

logging.basicConfig(level=logging.WARNING)

# Load .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b") 
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db_unified")
HR_COLLECTION = os.getenv("CHROMA_HR_COLLECTION", "resumes")
TRAVEL_COLLECTION = os.getenv("CHROMA_TRAVEL_COLLECTION", "travel-docs")

print("=" * 60)
print("  SUPERGEN DATA SEEDER")
print("=" * 60)


def get_embedding(text: str) -> list:
    """Get embedding from Ollama. Supports both new (/api/embed) and old (/api/embeddings) API."""
    # Try NEW Ollama API first (/api/embed)
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": text},
            timeout=120
        )
        if resp.status_code == 200:
            data = resp.json()
            # New API returns {"embeddings": [[...]]}
            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            # Some versions return {"embedding": [...]}
            return data.get("embedding", [])
    except Exception:
        pass

    # Fallback to OLD Ollama API (/api/embeddings)
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=120
        )
        if resp.status_code == 200:
            return resp.json().get("embedding", [])
        else:
            print(f"  [WARNING] Embedding failed on both /api/embed and /api/embeddings (HTTP {resp.status_code})")
            return []
    except Exception as e:
        print(f"  [ERROR] Embedding failed: {e}")
        return []

def warmup_ollama():
    """Warm up the Ollama model to prevent first-request timeouts."""
    print(f"\n[Ollama] Warming up model '{OLLAMA_EMBED_MODEL}' into memory... (This may take up to 2 minutes)")
    emb = get_embedding("warmup")
    if not emb:
        print(f"[Ollama] ERROR! Failed to load model '{OLLAMA_EMBED_MODEL}'. Please check if Ollama is running.")
        sys.exit(1)
    print(f"[Ollama] Model warmed up! Embedding dimension: {len(emb)}")

HR_CANDIDATES = [
    {
        "id": "candidate_001",
        "name": "Budi Santoso",
        "position": "Senior Python Developer",
        "skills": ["Python", "FastAPI", "Django", "PostgreSQL", "Docker"],
        "experience_years": 6,
        "education": "S1 Teknik Informatika, Universitas Indonesia",
        "location": "Jakarta",
        "available": True,
        "summary": (
            "Budi Santoso adalah Senior Python Developer dengan 6 tahun pengalaman. "
            "Ahli dalam membangun REST API menggunakan FastAPI dan Django. "
            "Berpengalaman dengan Docker dan PostgreSQL. Sertifikat AWS Cloud Practitioner."
        ),
    },
    {
        "id": "candidate_002",
        "name": "Siti Rahayu",
        "position": "Machine Learning Engineer",
        "skills": ["Python", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "SQL"],
        "experience_years": 4,
        "education": "S2 Ilmu Komputer, Universitas Gadjah Mada",
        "location": "Yogyakarta",
        "available": True,
        "summary": (
            "Siti Rahayu adalah Machine Learning Engineer dengan 4 tahun pengalaman "
            "dalam pengembangan model AI/ML untuk industri keuangan. "
            "Ahli deep learning TensorFlow, PyTorch, NLP, dan Computer Vision."
        ),
    },
    {
        "id": "candidate_003",
        "name": "Ahmad Fauzi",
        "position": "Full Stack Developer",
        "skills": ["JavaScript", "React", "Node.js", "Express", "MongoDB", "CSS"],
        "experience_years": 5,
        "education": "S1 Sistem Informasi, BINUS University",
        "location": "Bandung",
        "available": False,
        "summary": (
            "Ahmad Fauzi adalah Full Stack Developer 5 tahun. "
            "Ahli React.js dan Node.js/Express. "
            "Telah mengembangkan lebih dari 20 aplikasi web untuk industri fintech dan edtech."
        ),
    },
    {
        "id": "candidate_004",
        "name": "Dewi Lestari",
        "position": "Data Engineer",
        "skills": ["Python", "Apache Spark", "Airflow", "AWS", "GCP", "SQL", "Kafka"],
        "experience_years": 7,
        "education": "S1 Statistika, Institut Teknologi Bandung",
        "location": "Surabaya",
        "available": True,
        "summary": (
            "Dewi Lestari adalah Data Engineer senior 7 tahun. "
            "Membangun data pipeline besar dengan Apache Spark dan Airflow. "
            "Berpengalaman AWS dan GCP."
        ),
    },
    {
        "id": "candidate_005",
        "name": "Rizki Pratama",
        "position": "DevOps Engineer",
        "skills": ["Docker", "Kubernetes", "CI/CD", "Terraform", "Linux", "Python", "AWS"],
        "experience_years": 5,
        "education": "S1 Teknik Komputer, Universitas Brawijaya",
        "location": "Jakarta",
        "available": True,
        "summary": (
            "Rizki Pratama adalah DevOps Engineer 5 tahun dengan Docker dan Kubernetes. "
            "CI/CD via GitHub Actions dan Jenkins. Terraform untuk IaC."
        ),
    },
]

TRAVEL_LOCATIONS = [
    {
        "id": "loc_bali",
        "name": "Bali",
        "type": "Island",
        "country": "Indonesia",
        "best_time": "April-Oktober",
        "highlights": ["Pura Tanah Lot", "Ubud", "Tegallalang", "Seminyak Beach"],
        "description": (
            "Bali adalah pulau surga di Indonesia dengan pura Hindu, sawah berterasering, "
            "pantai eksotis, dan budaya seni. Surfing di Kuta, yoga di Ubud, diving di Amed, "
            "sunset di Uluwatu. Makanan: Babi Guling, Ayam Betutu. Mata uang: IDR."
        ),
    },
    {
        "id": "loc_jakarta",
        "name": "Jakarta",
        "type": "City",
        "country": "Indonesia",
        "best_time": "Mei-September",
        "highlights": ["Kota Tua", "Monas", "Ancol", "TMII"],
        "description": (
            "Jakarta adalah ibu kota Indonesia dan kota metropolitan terbesar di Asia Tenggara. "
            "Pusat bisnis, kuliner, dan hiburan. Kota Tua menawarkan sejarah kolonial Belanda. "
            "Kuliner: sate, gado-gado, kerak telor. Transportasi: MRT, LRT, Transjakarta."
        ),
    },
    {
        "id": "loc_yogyakarta",
        "name": "Yogyakarta",
        "type": "City",
        "country": "Indonesia",
        "best_time": "April-Oktober",
        "highlights": ["Borobudur", "Prambanan", "Malioboro", "Keraton"],
        "description": (
            "Yogyakarta adalah kota budaya dan pelajar. Dekat Borobudur (UNESCO) dan Prambanan. "
            "Terkenal dengan batik, wayang kulit, dan Gudeg. Malioboro adalah pusat oleh-oleh."
        ),
    },
    {
        "id": "loc_komodo",
        "name": "Pulau Komodo",
        "type": "National Park",
        "country": "Indonesia",
        "best_time": "April-Desember",
        "highlights": ["Komodo Dragon", "Pink Beach", "Padar Island", "Diving"],
        "description": (
            "Taman Nasional Komodo adalah Warisan Alam UNESCO di Nusa Tenggara Timur. "
            "Rumah Komodo, kadal terbesar di dunia. Pink Beach berpasir merah muda. "
            "Snorkeling dan diving kelas dunia. Akses via Labuan Bajo."
        ),
    },
    {
        "id": "loc_raja_ampat",
        "name": "Raja Ampat",
        "type": "Archipelago",
        "country": "Indonesia",
        "best_time": "Oktober-April",
        "highlights": ["Wayag", "Misool", "Pianemo", "Diving"],
        "description": (
            "Raja Ampat di Papua Barat adalah surga menyelam nomor satu di dunia. "
            "Biodiversitas laut tertinggi di bumi: 1.500+ spesies ikan, 600 karang. "
            "Pulau karst ikonik. Akses via Sorong."
        ),
    },
    {
        "id": "loc_vietnam",
        "name": "Vietnam",
        "type": "Country",
        "country": "Vietnam",
        "best_time": "Februari-April dan September-November",
        "highlights": ["Ha Long Bay", "Hoi An", "Ho Chi Minh City", "Hue", "Sapa"],
        "description": (
            "Vietnam adalah negara Asia Tenggara dengan kuliner lezat dan sejarah panjang. "
            "Ha Long Bay (UNESCO), Hoi An (kota kuno), Ho Chi Minh City (modern). "
            "Makanan: Pho, Banh Mi, Goi Cuon. Bahasa: Vietnam. Mata uang: VND."
        ),
    },
    {
        "id": "loc_nepal",
        "name": "Nepal",
        "type": "Country",
        "country": "Nepal",
        "best_time": "Maret-Mei dan September-November",
        "highlights": ["Everest Base Camp", "Annapurna", "Pokhara", "Kathmandu"],
        "description": (
            "Nepal adalah negara Himalaya dengan gunung tertinggi di dunia. "
            "Everest (8.849m), Annapurna Circuit. Kathmandu penuh candi Hindu dan Buddha. "
            "Destinasi trekking dunia. Visa on arrival untuk WNI."
        ),
    },
]

TRAVEL_RELATIONSHIPS = [
    ("loc_bali", "NEAR", "loc_komodo"),
    ("loc_yogyakarta", "NEAR", "loc_jakarta"),
    ("loc_bali", "POPULAR_WITH", "loc_raja_ampat"),
    ("loc_jakarta", "GATEWAY_TO", "loc_bali"),
    ("loc_jakarta", "GATEWAY_TO", "loc_yogyakarta"),
    ("loc_komodo", "GATEWAY_TO", "loc_raja_ampat"),
]


# ─────────────────────────────────────────────────────────────
# SEEDING FUNCTIONS
# ─────────────────────────────────────────────────────────────

def seed_chromadb():
    print("\n[ChromaDB] Initializing...")
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    hr_col = client.get_or_create_collection(name=HR_COLLECTION, metadata={"hnsw:space": "cosine"})
    travel_col = client.get_or_create_collection(name=TRAVEL_COLLECTION, metadata={"hnsw:space": "cosine"})

    # --- HR Candidates ---
    print(f"[ChromaDB] Seeding {len(HR_CANDIDATES)} HR candidates...")
    for c in HR_CANDIDATES:
        doc = (
            f"Nama: {c['name']}\nPosisi: {c['position']}\n"
            f"Skills: {', '.join(c['skills'])}\nPengalaman: {c['experience_years']} tahun\n"
            f"Pendidikan: {c['education']}\nLokasi: {c['location']}\n"
            f"Tersedia: {'Ya' if c['available'] else 'Tidak'}\n\nRingkasan: {c['summary']}"
        )
        emb = get_embedding(doc)
        meta = {"name": c["name"], "position": c["position"],
                "skills": ", ".join(c["skills"]), "location": c["location"]}

        if emb:
            try:
                hr_col.add(ids=[c["id"]], documents=[doc], embeddings=[emb], metadatas=[meta])
            except Exception:
                hr_col.update(ids=[c["id"]], documents=[doc], embeddings=[emb], metadatas=[meta])
        else:
            print(f"  [ERROR] Skipped {c['name']} because embedding failed.")

        print(f"  [OK] {c['name']} ({c['position']})")

    # --- Travel Locations ---
    print(f"\n[ChromaDB] Seeding {len(TRAVEL_LOCATIONS)} travel locations...")
    for loc in TRAVEL_LOCATIONS:
        doc = (
            f"Destinasi: {loc['name']}\nTipe: {loc['type']}\nNegara: {loc['country']}\n"
            f"Waktu Terbaik: {loc['best_time']}\nHighlight: {', '.join(loc['highlights'])}\n\n"
            f"Deskripsi: {loc['description']}"
        )
        emb = get_embedding(doc)
        meta = {"name": loc["name"], "type": loc["type"],
                "country": loc["country"], "title": loc["name"]}

        if emb:
            try:
                travel_col.add(ids=[loc["id"]], documents=[doc], embeddings=[emb], metadatas=[meta])
            except Exception:
                travel_col.update(ids=[loc["id"]], documents=[doc], embeddings=[emb], metadatas=[meta])
        else:
            print(f"  [ERROR] Skipped {loc['name']} because embedding failed.")

        print(f"  [OK] {loc['name']} ({loc['country']})")

    print(f"\n[ChromaDB] Total: {hr_col.count()} resumes | {travel_col.count()} travel docs")


def seed_neo4j():
    print("\n[Neo4j] Connecting...")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except Exception as e:
        print(f"[Neo4j] SKIP - cannot connect: {e}")
        return

    with driver.session() as session:
        session.run("MATCH (n:Candidate) DETACH DELETE n")
        session.run("MATCH (n:Location) DETACH DELETE n")
        session.run("MATCH (n:Skill) DETACH DELETE n")
        print("[Neo4j] Old data cleared.")

        print(f"[Neo4j] Creating {len(HR_CANDIDATES)} Candidate + Skill nodes...")
        for c in HR_CANDIDATES:
            session.run("""
                MERGE (c:Candidate {id: $id})
                SET c.name=$name, c.position=$pos, c.experience_years=$exp,
                    c.education=$edu, c.location=$loc, c.available=$avail, c.summary=$summary
            """, {"id": c["id"], "name": c["name"], "pos": c["position"],
                  "exp": c["experience_years"], "edu": c["education"],
                  "loc": c["location"], "avail": c["available"], "summary": c["summary"]})
            for skill in c["skills"]:
                session.run("""
                    MERGE (s:Skill {name: $skill})
                    WITH s MATCH (c:Candidate {id: $cid}) MERGE (c)-[:HAS_SKILL]->(s)
                """, {"skill": skill, "cid": c["id"]})
            print(f"  [OK] {c['name']} + {len(c['skills'])} skills")

        print(f"\n[Neo4j] Creating {len(TRAVEL_LOCATIONS)} Location nodes...")
        for loc in TRAVEL_LOCATIONS:
            session.run("""
                MERGE (l:Location {id: $id})
                SET l.name=$name, l.type=$type, l.country=$country,
                    l.description=$desc, l.best_time=$best_time
            """, {"id": loc["id"], "name": loc["name"], "type": loc["type"],
                  "country": loc["country"], "desc": loc["description"],
                  "best_time": loc["best_time"]})
            print(f"  [OK] {loc['name']} ({loc['type']}, {loc['country']})")

        print(f"\n[Neo4j] Creating {len(TRAVEL_RELATIONSHIPS)} relationships...")
        for src, rel_type, dst in TRAVEL_RELATIONSHIPS:
            session.run(f"""
                MATCH (a:Location {{id: $src}}), (b:Location {{id: $dst}})
                MERGE (a)-[:{rel_type}]->(b)
            """, {"src": src, "dst": dst})
            print(f"  [OK] {src} -{rel_type}-> {dst}")

        # Full-text index for location search
        try:
            session.run("DROP INDEX locationFullTextIndex IF EXISTS")
            session.run("""
                CREATE FULLTEXT INDEX locationFullTextIndex
                FOR (n:Location) ON EACH [n.name, n.description, n.type, n.country]
            """)
            print("\n[Neo4j] Full-text index created.")
        except Exception as e:
            print(f"[Neo4j] Index warning (may already exist): {e}")

        result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt ORDER BY cnt DESC")
        print("\n[Neo4j] Final node counts:")
        for row in result:
            print(f"  {row['label']}: {row['cnt']}")

    driver.close()


# ─────────────────────────────────────────────────────────────
def main():
    # Check Ollama reachability and warmup
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    except Exception:
        print("\n[Ollama] ERROR! Ollama API not reachable. Please start 'ollama serve'.")
        sys.exit(1)
        
    warmup_ollama()

    seed_chromadb()
    seed_neo4j()

    print("\n" + "=" * 60)
    print("  SEEDING COMPLETE!")
    print("=" * 60)
    print("  > Data loaded into ChromaDB and Neo4j.")
    print("  > Jalankan CLI dengan Python 3.10:")
    print("    .\\HybridKnowledge\\venv_310\\Scripts\\python.exe cli.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
