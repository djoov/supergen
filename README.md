# SuperGen Modular (Unified AI System)

System **SuperGen Modular** adalah platform AI berbasis arsitektur **Hybrid RAG (Retrieval-Augmented Generation)** yang memadukan kekuatan Vector Database (ChromaDB) dan Knowledge Graph (Neo4j). Sistem ini mendukung *Multi-Agent Processing*, memungkinkan spesialisasi AI untuk HR (Pencarian Kandidat) dan Travel (Rencana Perjalanan).

---

## 🚀 Fitur Utama

1. **Hybrid Knowledge Retrieval**
   Menggabungkan Vector Semantic Search (ChromaDB) untuk konteks dokumen kasar, dan Graph Structural Search (Neo4j) untuk melacak relasi entitas/dokumen.
2. **Multi-Agent Architecture (AutoGen)**
   Mendukung agen *standalone* untuk query cepat dan agen berbasis *AutoGen Group Chat* (Planner, Local Expert, Language Tips) untuk pembuatan itinerary kompleks.
3. **Advanced Conflict Resolution Engine**
   Mendeteksi dan menyelesaikan konflik data antar dokumen lama vs baru secara otomatis berdasarkan metadata waktu dan versi (tanpa perlu menghapus data lama).

---

## 🛠️ Persyaratan Sistem

- **Python 3.14** (direkomendasikan di environment `venv`)
- **Ollama** berjalan secara lokal dengan model `llama3.1:8b`
- **Neo4j Desktop / Community Edition** berjalan secara lokal (default port: 7687)

---

## ⚙️ Menjalankan Sistem

1. Aktifkan Virtual Environment:
   ```bash
   .\venv\Scripts\activate
   ```
2. Jalankan antarmuka interaktif CLI:
   ```bash
   python cli.py
   ```

Pilih agen di menu yang tersedia, misalnya:
- `[1]` **HR Assistant** (tanya: *"siapa kandidat devops yang kita punya?"*)
- `[2]` **Travel (Simple)** (tanya: *"apa rekomendasi wisata di Vietnam?"*)
- `[3]` **Travel (AutoGen)** (tanya: *"buatkan itinerary 3 hari di Tokyo"*)

---

## 🛡️ Conflict Resolution Engine

Salah satu keunggulan utama sistem ini adalah mampu **secara otomatis menyingkirkan informasi usang** tanpa menghapusnya dari database. 

### Bagaimana AI Menentukan Kebenaran?
Ketika AI menemukan dua dokumen yang bertentangan, ia tidak hanya menggunakan *Similarity Score* (kemiripan teks), namun melakukan **Weighted Re-ranking** (Penilaian Ulang Berbobot) berdasarkan:
- **Recency (30%):** Seberapa baru dokumen tersebut diunggah.
- **Version (15%):** Nomor versi dokumen.
- **Source (5%):** Tingkat kepercayaan sumber (contoh: 'kebijakan_resmi' mengalahkan 'email').
- **Supersedes Graph:** Relasi `[:SUPERSEDES]` di Neo4j langsung mencoret dokumen lama dari konteks LLM.

### Cara Mengupdate Data Baru (Data Seeding)
Gunakan `data_loader.py` untuk mengimpor dokumen JSON ke dalam ChromaDB & Neo4j. Script ini **otomatis** akan menyematkan *Temporal Metadata* (tanggal upload).

**1. Import Data Standar**
```bash
python data_loader.py data_kandidat_baru.json --target hr
```

**2. Import dengan Mendeklarasikan Penggantian Dokumen Lama (Supersedes)**
Jika Anda punya dokumen baru (misal kebijakan 2026) yang menggugurkan kebjakan lama:
```bash
python data_loader.py kebijakan_2026.json --target hr --version 2 --source-type kebijakan_resmi --supersedes "id_dokumen_kebijakan_lama_2020"
```
*Dengan perintah di atas, Neo4j akan mencoret kebijakan 2020, dan AI hanya akan menjawab berdasarkan kebijakan 2026.*

### Mengetes Kemampuan Conflict Resolution
Anda dapat menyimulasikan konflik dokumen dengan menjalankan *mock test* yang tersedia:
```bash
python test_conflict.py
```
*Script ini mensimulasikan upload 2 dokumen ("Kebijakan Gaji Lama" vs "Kebijakan Gaji 2026") dan membuktikan bahwa AI secara cerdas selalu memenangkan dokumen tahun 2026, walau dokumen lama mungkin memiliki skor kemiripan vektor yang lebih tinggi.*

---
*Developed with Hybrid AI Architecture by the SuperGen Team.*
