import sys
from fpdf import FPDF

content = """SuperGen Modular AI - Fitur Re-ranking (Penilaian Ulang)

1. MENGAPA KITA MEMBUTUHKAN RE-RANKING?
Vector Database seperti ChromaDB sangat cepat menemukan dokumen berdasarkan kecocokan teks semantik (Semantic Similarity), namun mereka BUTA TERHADAP WAKTU DAN KEBENARAN. 

Jika ada dua versi dokumen:
- Dokumen A: SOP Cuti Karyawan 2020
- Dokumen B: Revisi SOP Cuti Karyawan 2026

Mesin pencari standar bisa saja memberikan Dokumen A kepada AI hanya karena kata-katanya kebetulan lebih "cocok" dengan pertanyaan user. Akibatnya, AI akan membaca Dokumen A dan memberikan jawaban yang SALAH atau HALUSINASI DATA. 

Fitur Re-ranking ada untuk mengoreksi bias pencarian standar ini.

2. BAGAIMANA RE-RANKING BEKERJA (ANALOGI PUSTAKAWAN)
Bayangkan Anda adalah manajer, dan Anda memiliki dua asisten:
- Asisten 1 (Vector Search): Bertugas mencari buku tercepat berdasarkan kemiripan judul.
- Asisten 2 (Re-ranker): Bertugas mensortir ulang buku hasil pencarian Asisten 1.

Setelah Asisten 1 membawa Buku A (2020) dan Buku B (2026), Asisten 2 akan mencegatnya. Asisten 2 menggunakan fungsi matematis untuk menilai ulang buku-buku tersebut berdasarkan Metadata (Data historis):

A. Recency (Bobot 30%): Seberapa baru dokumen di-upload? Dok. 2026 mendapat nilai sempurna (1.0), sedangkan 2020 nilainya turun (0.1).
B. Version (Bobot 15%): Apakah ini rilis versi terbaru?
C. Source (Bobot 5%): Apakah sumber kredibel (Kebijakan Resmi vs Email)?

Setelah dijumlahkan dengan skor kemiripan asli, Asisten 2 akan DENGAN PAKSA menurunkan peringkat Buku A (2020) dan mengorbitkan Buku B (2026) menjadi posisi teratas.

3. MANFAAT UNTUK ENTERPRISE AI
Berkat perlindungan Re-ranking Engine ini, saat AI (LLM / Llama / Qwen) mengumpulkan jawaban untuk Anda, ia HANYA BISA MEMBACA FAKTA TERBARU.

Skor akhir AI SuperGen bukan lagi sebatas "kemiripan kata", melainkan hasil "Critical Thinking" (Analisa Kritis) otomatis layaknya analis manusia yang mengecek validitas tanggal dan versi sebuah kebijakan sebelum mempublikasikannya.
"""

pdf = FPDF()
pdf.add_page()
pdf.set_font("helvetica", size=11)
# Bersihkan karakter non-latin1 agar tidak error di fpdf default font
clean_text = content.encode('latin-1', 'replace').decode('latin-1')
pdf.multi_cell(0, 6, txt=clean_text)
out_file = "Laporan_Fitur_Reranking.pdf"
pdf.output(out_file)
print(f"Created {out_file}")
