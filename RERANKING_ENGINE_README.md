# SuperGen Advanced Re-ranking Engine
*Technical Documentation & Architectural Decisions*

Dokumen ini menjelaskan secara teknis dan matematis mengapa SuperGen Modular menggunakan kustom **Conflict Resolver (Re-ranker)** di atas ChromaDB, dan bagaimana algoritma ini mencegah *LLM Hallucination* (halusinasi AI) pada skenario dokumen temporal (dokumen yang berubah seiring waktu).

---

## 1. The Core Problem: Keterbatasan Semantic Vector Search
Sistem RAG (Retrieval-Augmented Generation) konvensional menggunakan Vector Database (seperti ChromaDB) untuk mencari dokumen yang relevan dengan '*query*' pengguna. 

**Bagaimana Vector DB Bekerja:**
ChromaDB menggunakan model *embedding* (misal: `Ollama llama3.1`) untuk mengubah teks menjadi deretan angka (vektor n-dimensi). Saat user bertanya, sistem menghitung **Cosine Similarity** (Jarak Kosinus) antara vektor pertanyaan dan vektor dokumen. Semakin dekat jaraknya, semakin relevan dokumen tersebut.

### 🔴 The Vulnerability (Titik Kelemahan)
Vector DB murni bersifat **A-Temporal (Buta Waktu)** dan **A-Factual (Buta Fakta)**.
Mereka hanya mengukur *seberapa mirip struktur bahasanya*, bukan *apakah informasinya masih berlaku saat ini*.

**Skenario Bencana:**
- **Dokumen A:** "Gaji WRE dibayarkan setiap tanggal 25." *(Diunggah 5 tahun lalu)*
- **Dokumen B:** "Mulai tahun 2026, pembayaran instruktur WRE diubah menjadi rilis tanggal 1." *(Diunggah hari ini)*

Jika user bertanya: *"Kapan gaji WRE dibayarkan?"*, **Dokumen A akan selalu menang dan mendapat skor Cosine Similarity yang lebih tinggi** karena sintaksis dan panjang kalimatnya lebih identik dengan pertanyaan (kalimatnya langsung *to the point*).

Jika hanya mengandalkan ChromaDB, sistem RAG akan menyerahkan Dokumen A ke LLM. LLM kemudian dengan yakin menjawab *"Gaji WRE dibayarkan tanggal 25"*. Ini adalah kegagalan fatal (*Fatal Hallucination*) dalam aplikasi Enterprise berskala produksi.

---

## 2. SuperGen Solution: The Re-ranking Engine
Untuk mengatasi masalah ini, SuperGen secara arsitektural mencegat (*intercept*) *output* mentah dari ChromaDB sebelum disalurkan ke LLM.

Proses intersepsi ini ditangani oleh modul `core/conflict_resolver.py`.

### A. Temporal Metadata Injection (`data_loader.py`)
Sebagai prasyarat, setiap data yang masuk ke ChromaDB disuntikkan metadata absolut:
```json
{
  "uploaded_at": "2026-03-16T10:00:00Z", // Waktu injeksi (ISO-8601 UTC)
  "version": "2",                        // Iterasi dokumen
  "source_type": "kebijakan_resmi"       // Hierarki otoritas
}
```

### B. Mathematical Re-ranking (Skoring Ulang)
`conflict_resolver.py` tidak membuang skor *Cosine Similarity* dari ChromaDB, melainkan *menurunkannya* menjadi hanya salah satu dari sekian variabel matematika.

Algoritma menghitung **Final Score** menggunakan formula regresi terbobot:
```python
final_score = (
    (sim_score * bobot_similarity) +
    (recency_score * bobot_recency) +
    (version_score * bobot_version) +
    (source_score * bobot_source)
)
```

**Default Weights (Distribusi Bobot):**
- **Similarity (50%):** Kemiripan makna semantik (dari ChromaDB).
- **Recency (30%):** Usia dokumen (Dihitung dengan peluruhan eksponensial/linear, di mana usia 0 hari = 1.0, usia >1 tahun = mendekati 0.0).
- **Version (15%):** Dokumen dengan iterasi yang lebih tinggi (V.2) mengalahkan iterasi sebelumnya (V.1).
- **Source (5%):** Klasifikasi sumber (Dokumen Resmi `1.0` vs Catatan Personal `0.4`).

### C. Pembuktian Algoritma pada Skenario
Menggunakan Skenario Bencana di atas, kalkulator Re-ranker akan memprosesnya sbb:

**Dokumen A (2020 - Lama):**
- Vector Similarity: `0.75` (Tinggi)
- Recency Score: `0.05` (Rendah - 5 tahun lalu)
- Version Score: `0.1` (V1)
- Source Score: `1.0` (Resmi)
- **FINAL SCORE = (0.75*0.5) + (0.05*0.3) + (0.1*0.15) + (1.0*0.05) = `0.455`**

**Dokumen B (2026 - Baru):**
- Vector Similarity: `0.62` (Lebih Rendah)
- Recency Score: `1.00` (Sempurna - Baru diunggah)
- Version Score: `0.5` (V2)
- Source Score: `1.0` (Resmi)
- **FINAL SCORE = (0.62*0.5) + (1.00*0.3) + (0.5*0.15) + (1.0*0.05) = `0.735`**

**Hasil Akhir:** Dokumen B (Baru) menang telak dengan skor `0.735` vs `0.455`. Sistem secara paksa *me-rerank* (menukar posisi) Dokumen B ke Ranking 1. LLM dijamin hanya akan mengonsumsi Dokumen B.

---

## 3. The 3-Layer Defense Architecture
Untuk memastikan integritas 100%, Re-ranking Engine bukan satu-satunya pertahanan. SuperGen menggunakan 3 lapisan proteksi secara berbayang (Cascading Defense):

1. **Layer 1: Neo4j Explicit Resolving (Hard Filter).** 
   Jika administrator secara eksplisit mendeklarasikan `SUPERSEDES` (Dokumen B menimpa Dokumen A) saat *seeding*, `conflict_resolver.py` akan langsung **memvonis mati** Dokumen A. Dokumen A tidak akan diikutsertakan dalam Re-ranking sama sekali.
2. **Layer 2: Implicit Re-ranking (Soft Filter).** 
   Jika administrator lupa mendeklarasikan `SUPERSEDES`, Layer 2 (seperti penjelasan rumus matematis di atas) akan otomatis memenangkan Dokumen B secara implisit berdasarkan *Recency* dan *Version*.
3. **Layer 3: LLM Context Instructions.**
   Sebagai lapisan pamungkas, *System Prompt* pada `hr_agent.py` dan `travel_agent.py` diinstruksikan secara tekstual: *"Jika Anda melihat sekumpulan dokumen yang saling bertentangan dalam konteks ini, selalulah prioritaskan yang memiliki metadata temporal paling baru."*

---

## 4. Kesimpulan & Manfaat Enterprise
Sistem berbasis Vector semata tidak bisa dipercaya untuk menangani dokumen dinamis (seperti SOP, Data Pegawai, Skema Pajak). 

Dengan mengimplementasikan **Advanced Re-ranking Engine**, SuperGen Modular memiliki *Critical Thinking* logis tersendiri *sebelum* diserahkan ke LLM, menjaga arsitektur RAG ini aman untuk diproduksi (Production-Ready) dan mempertahankan sifat **Immutable Data** (data lama tidak perlu dihapus, dibiarkan saja menumpuk untuk kebutuhan audit log).
