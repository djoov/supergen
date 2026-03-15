"""
Clean ChromaDB migration script.
MUST be run with the SAME Python environment as cli.py (venv).
Usage: .\venv\Scripts\python.exe migrate_chroma.py
"""
import os, json, sys, requests
from dotenv import load_dotenv
load_dotenv('.env')

import chromadb
print(f"Python: {sys.version}")
print(f"ChromaDB version: {chromadb.__version__}")
print(f"Persist dir: {os.getenv('CHROMA_PERSIST_DIR')}")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

def get_embedding(text):
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_MODEL, "prompt": text},
            timeout=120
        )
        if resp.status_code == 200:
            return resp.json().get("embedding", [])
    except Exception as e:
        print(f"  [WARN] Embedding timeout/error: {e}")
    return []

# Initialize clean client
client = chromadb.PersistentClient(path=os.getenv('CHROMA_PERSIST_DIR'))

# Delete existing collections if any
for name in ['resumes', 'travel-docs']:
    try:
        client.delete_collection(name)
        print(f"Deleted old collection: {name}")
    except:
        pass

hr_col = client.create_collection(name='resumes', metadata={'hnsw:space': 'cosine'})
travel_col = client.create_collection(name='travel-docs', metadata={'hnsw:space': 'cosine'})

# ─── HR CANDIDATES ───
HR_CANDIDATES = [
    {"id": "candidate_001", "name": "Budi Santoso", "position": "Senior Python Developer",
     "skills": ["Python", "FastAPI", "Django", "PostgreSQL", "Docker"], "experience_years": 6,
     "education": "S1 Teknik Informatika, Universitas Indonesia", "location": "Jakarta", "available": True,
     "summary": "Budi Santoso adalah Senior Python Developer dengan 6 tahun pengalaman. Ahli dalam membangun REST API menggunakan FastAPI dan Django. Berpengalaman dengan Docker dan PostgreSQL. Sertifikat AWS Cloud Practitioner."},
    {"id": "candidate_002", "name": "Siti Rahayu", "position": "Machine Learning Engineer",
     "skills": ["Python", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "SQL"], "experience_years": 4,
     "education": "S2 Ilmu Komputer, Universitas Gadjah Mada", "location": "Yogyakarta", "available": True,
     "summary": "Siti Rahayu adalah Machine Learning Engineer dengan 4 tahun pengalaman dalam pengembangan model AI/ML untuk industri keuangan. Ahli deep learning TensorFlow, PyTorch, NLP, dan Computer Vision."},
    {"id": "candidate_003", "name": "Ahmad Fauzi", "position": "Full Stack Developer",
     "skills": ["JavaScript", "React", "Node.js", "Express", "MongoDB", "CSS"], "experience_years": 5,
     "education": "S1 Sistem Informasi, BINUS University", "location": "Bandung", "available": False,
     "summary": "Ahmad Fauzi adalah Full Stack Developer 5 tahun. Ahli React.js dan Node.js/Express. Telah mengembangkan lebih dari 20 aplikasi web untuk industri fintech dan edtech."},
    {"id": "candidate_004", "name": "Dewi Lestari", "position": "Data Engineer",
     "skills": ["Python", "Apache Spark", "Airflow", "AWS", "GCP", "SQL", "Kafka"], "experience_years": 7,
     "education": "S1 Statistika, Institut Teknologi Bandung", "location": "Surabaya", "available": True,
     "summary": "Dewi Lestari adalah Data Engineer senior 7 tahun. Membangun data pipeline besar dengan Apache Spark dan Airflow. Berpengalaman AWS dan GCP."},
    {"id": "candidate_005", "name": "Rizki Pratama", "position": "DevOps Engineer",
     "skills": ["Docker", "Kubernetes", "CI/CD", "Terraform", "Linux", "Python", "AWS"], "experience_years": 5,
     "education": "S1 Teknik Komputer, Universitas Brawijaya", "location": "Jakarta", "available": True,
     "summary": "Rizki Pratama adalah DevOps Engineer 5 tahun dengan Docker dan Kubernetes. CI/CD via GitHub Actions dan Jenkins. Terraform untuk IaC."},
]

print("\n=== Seeding HR Candidates ===")
for c in HR_CANDIDATES:
    doc = f"Nama: {c['name']}\nPosisi: {c['position']}\nSkills: {', '.join(c['skills'])}\nPengalaman: {c['experience_years']} tahun\nPendidikan: {c['education']}\nLokasi: {c['location']}\nTersedia: {'Ya' if c['available'] else 'Tidak'}\n\nRingkasan: {c['summary']}"
    emb = get_embedding(doc)
    meta = {'name': str(c['name']), 'position': str(c['position']),
            'skills': ', '.join(c['skills']), 'location': str(c['location'])}
    if emb:
        hr_col.add(ids=[c['id']], documents=[doc], embeddings=[emb], metadatas=[meta])
    else:
        hr_col.add(ids=[c['id']], documents=[doc], metadatas=[meta])
    print(f"  [OK] {c['name']}")

# ─── BASIC TRAVEL LOCATIONS ───
TRAVEL_LOCATIONS = [
    {"id": "loc_bali", "name": "Bali", "type": "Island", "country": "Indonesia",
     "best_time": "April-Oktober", "highlights": ["Pura Tanah Lot", "Ubud", "Tegallalang", "Seminyak Beach"],
     "description": "Bali adalah pulau surga di Indonesia dengan pura Hindu, sawah berterasering, pantai eksotis, dan budaya seni. Surfing di Kuta, yoga di Ubud, diving di Amed, sunset di Uluwatu. Makanan: Babi Guling, Ayam Betutu. Mata uang: IDR."},
    {"id": "loc_jakarta", "name": "Jakarta", "type": "City", "country": "Indonesia",
     "best_time": "Mei-September", "highlights": ["Kota Tua", "Monas", "Ancol", "TMII"],
     "description": "Jakarta adalah ibu kota Indonesia dan kota metropolitan terbesar di Asia Tenggara. Pusat bisnis, kuliner, dan hiburan. Kota Tua menawarkan sejarah kolonial Belanda."},
    {"id": "loc_yogyakarta", "name": "Yogyakarta", "type": "City", "country": "Indonesia",
     "best_time": "April-Oktober", "highlights": ["Borobudur", "Prambanan", "Malioboro", "Keraton"],
     "description": "Yogyakarta adalah kota budaya dan pelajar. Dekat Borobudur (UNESCO) dan Prambanan. Terkenal dengan batik, wayang kulit, dan Gudeg."},
    {"id": "loc_vietnam", "name": "Vietnam", "type": "Country", "country": "Vietnam",
     "best_time": "Februari-April dan September-November", "highlights": ["Ha Long Bay", "Hoi An", "Ho Chi Minh City", "Hue", "Sapa"],
     "description": "Vietnam adalah negara Asia Tenggara dengan kuliner lezat dan sejarah panjang. Ha Long Bay (UNESCO), Hoi An (kota kuno), Ho Chi Minh City (modern). Makanan: Pho, Banh Mi, Goi Cuon."},
]

print("\n=== Seeding Basic Travel ===")
for loc in TRAVEL_LOCATIONS:
    doc = f"Destinasi: {loc['name']}\nTipe: {loc['type']}\nNegara: {loc['country']}\nWaktu Terbaik: {loc['best_time']}\nHighlight: {', '.join(loc['highlights'])}\n\nDeskripsi: {loc['description']}"
    emb = get_embedding(doc)
    meta = {'name': str(loc['name']), 'type': str(loc['type']),
            'country': str(loc['country']), 'title': str(loc['name'])}
    if emb:
        travel_col.add(ids=[loc['id']], documents=[doc], embeddings=[emb], metadatas=[meta])
    else:
        travel_col.add(ids=[loc['id']], documents=[doc], metadatas=[meta])
    print(f"  [OK] {loc['name']}")

# ─── VIETNAM TRAVEL DATASET (360 items) ───
vn_path = 'HybridKnowledge/vietnam_travel_dataset.json'
if os.path.exists(vn_path):
    with open(vn_path, 'r', encoding='utf-8') as f:
        vn_data = json.load(f)
    print(f"\n=== Seeding {len(vn_data)} Vietnam Travel Items ===")
    for i, item in enumerate(vn_data):
        t = item.get('type', '')
        if t == 'Location':
            doc = f"Nama Destinasi: {item.get('name', '')}\nTipe: {item.get('location_type', '')}\nDeskripsi: {item.get('description', '')}\nRegion: {item.get('region', '')}\nWaktu Terbaik: {item.get('best_time_to_visit', '')}\nTags: {', '.join(item.get('tags', []))}"
        elif t == 'Hotel':
            doc = f"Nama Hotel: {item.get('name', '')}\nBintang: {item.get('star_rating', '')}\nKota: {item.get('city', '')}\nDeskripsi: {item.get('description', '')}\nAmenities: {', '.join(item.get('amenities', []))}\nDekat dengan: {', '.join(item.get('nearby_attractions', []))}"
        elif t == 'Activity':
            doc = f"Nama Aktivitas: {item.get('name', '')}\nDurasi: {item.get('duration', '')}\nLokasi: {item.get('location', '')}\nDeskripsi: {item.get('description', '')}\nTermasuk: {', '.join(item.get('included', []))}\nHarga: {item.get('price_estimate', '')}"
        else:
            doc = json.dumps(item, ensure_ascii=False)
        
        emb = get_embedding(doc)
        meta = {
            'name': str(item.get('name', ''))[:200],
            'type': str(item.get('type', ''))[:100],
            'country': 'Vietnam',
            'location': str(item.get('city', item.get('location', item.get('region', ''))))[:100],
            'source': 'vietnam_travel_dataset'
        }
        if emb:
            travel_col.add(ids=[str(item.get('id', f'vietnam_{i}'))], documents=[doc], embeddings=[emb], metadatas=[meta])
        else:
            travel_col.add(ids=[str(item.get('id', f'vietnam_{i}'))], documents=[doc], metadatas=[meta])
        if i % 10 == 0:
            print(f"  [{i}/{len(vn_data)}] {item.get('name', '')[:40]}")
    
    print(f"  Vietnam import complete!")
else:
    print(f"\n[SKIP] Vietnam dataset not found at {vn_path}")

print(f"\n{'='*60}")
print(f"  MIGRATION COMPLETE!")
print(f"  HR: {hr_col.count()} documents")
print(f"  Travel: {travel_col.count()} documents")
print(f"{'='*60}")
