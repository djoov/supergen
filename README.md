# SuperGen Modular (Unified AI System)

Sistem **SuperGen Modular** adalah platform AI tingkat Enterprise berbasis **Hybrid RAG (Retrieval-Augmented Generation)**. Sistem ini memadukan kekuatan Semantic Search dari Vector Database (ChromaDB) dan Structural Search dari Knowledge Graph (Neo4j). 

Dilengkapi dengan **Advanced Conflict Resolution** dan **Multi-Agent Architecture (AutoGen)**, SuperGen mampu mensortir data usang secara otomatis dan membiarkan beberapa AI Agent berdiskusi untuk menghasilkan jawaban terbaik (contoh: merumuskan Itinerary Travel kompleks).

---

## 🚀 Fitur Utama
1. **Hybrid Knowledge Retrieval:** Vector (ChromaDB) + Graph (Neo4j).
2. **Multi-Agent Architecture:** Ditenagai oleh AutoGen Microsoft.
3. **Advanced Conflict Resolution Engine:** Otomatis "mencoret" data lama yang bertentangan dengan kebijakan baru menggunakan *Temporal Metadata* & *Neo4j SUPERSEDES relationship*. AI Anda dijamin bebas dari *outdated hallucination*.
4. **Local LLM:** Ditenagai sepenuhnya oleh Ollama (Llama 3) untuk menjaga privasi data perusahaan 100%.

---

## 🛠️ Prasyarat & Instalasi (Step-by-Step)

Sistem ini berjalan sepenuhnya di lokal. Anda membutuhkan 3 komponen utama: **Python**, **Ollama**, dan **Neo4j**.

### Langkah 1: Install & Jalankan Ollama (Local LLM & Embedding)
1. Unduh Ollama dari [ollama.com](https://ollama.com/) dan install.
2. Buka terminal/command prompt dan jalankan perintah penarikan model:
   ```bash
   # Model untuk Agent reasoning (wajib)
   ollama pull llama3.1:8b
   
   # Note: Embedding saat ini menggunakan endpoint /api/embeddings dari model yang sama
   ```
3. Pastikan Ollama berjalan di *background* (default port: `11434`).

### Langkah 2: Setup Neo4j (Knowledge Graph) via Docker
Sangat disarankan menggunakan Docker untuk Neo4j agar bersih dan terisolasi.
1. Pastikan Docker Desktop / Docker Engine sudah terinstall.
2. Jalankan Neo4j container dengan perintah berikut:
   ```bash
   docker run -d \
     --name supergen-neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password123 \
     -v $PWD/neo4j_data:/data \
     neo4j:latest
   ```
3. Buka browser ke `http://localhost:7474` untuk melihat tampilan visual Graph Database (Login: `neo4j` / `password123`).

### Langkah 3: Setup Environment Python
1. Clone repositori ini.
2. Buat Virtual Environment Python (direkomendasikan versi 3.10 - 3.14):
   ```bash
   python -m venv venv
   ```
3. Aktifkan Virtual Environment:
   * Windows: `.\venv\Scripts\activate`
   * Mac/Linux: `source venv/bin/activate`
4. Install semua dependensi:
   ```bash
   pip install -r requirements.txt
   ```

### Langkah 4: Konfigurasi Environment Variables (`.env`)
Salin file `.env.example` menjadi `.env` (atau buat file `.env` baru di folder utama) dan pastikan isinya:
```ini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# Folder lokal untuk menyimpan Vector DB
CHROMA_PERSIST_DIR=./chroma_v4
CHROMA_HR_COLLECTION=resumes
CHROMA_TRAVEL_COLLECTION=travel-docs
```

---

## ⚙️ Memasukkan Data (Data Seeding)

Sebelum AI bisa digunakan, Anda harus memasukkan (ingest) data ke ChromaDB dan Neo4j menggunakan skrip `data_loader.py`.

### 1. Import Data Standar
```bash
# Untuk HR
python data_loader.py data_kandidat_baru.json --target hr

# Untuk Travel
python data_loader.py data_travel_vietnam.json --target travel
```

### 2. Import dengan Conflict Resolution (Mengganti Kebijakan Lama)
Keunggulan SuperGen adalah Anda tidak perlu menghapus file lama. Cukup beritahu sistem bahwa file baru ini "meralat" (*supersedes*) ID dokumen yang lama.
```bash
python data_loader.py kebijakan_2026.json --target hr --version 2 --source-type kebijakan_resmi --supersedes "id_dokumen_kebijakan_2020"
```
*Dengan perintah di atas, Neo4j akan menggaris-merahi dokumen lama. Saat AI ditanya, IA akan 100% menggunakan kebijakan 2026 yang baru.*

---

## 💻 Penggunaan Sistem (CLI)

Gunakan *Antarmuka Command Line* (CLI) untuk berinteraksi dengan agen AI yang telah Anda bangun.
```bash
python cli.py
```
Anda akan disambut oleh menu pilihan Agen:
* `[1]` **HR Assistant:** Agent untuk merangkum dan mencari kandidat atau kebijakan internal.
* `[2]` **Travel (Simple):** Agent Q&A perjalanan yang cepat.
* `[3]` **Travel (AutoGen - Multi Agent):** Memanggil 4 agen terpisah (*Planner, Local Guide, Language Expert, Summarizer*) yang akan berdiskusi satu sama lain untuk merancang itinerary liburan komprehensif untuk Anda.

---

## 🧪 Testing Conflict Resolution
Ingin melihat bagaimana AI pintar kita memilah fakta lama vs fakta baru secara real-time? Jalankan skrip tes berikut:
```bash
python test_conflict.py
```
*Skrip ini memasukkan dua data yang saling bertentangan secara sengaja, lalu membuktikan secara matematis bahwa Document Reranking (bobot usia dokumen + versi dokumen) berhasil menangkal halusinasi AI standar.*
