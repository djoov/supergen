import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

#Bobot default untuk re-ranking
DEFAULT_WEIGHTS = {
    "similarity": 0.5,    # Skor vektor similarity
    "recency": 0.3,       # Seberapa baru dokumen
    "version": 0.15,      # Nomor versi
    "source": 0.05,       # Tipe sumber (resmi > informal)
}

# Bobot berdasarkan tipe sumber dokumen
SOURCE_TYPE_SCORES = {
    "kebijakan_resmi": 1.0,
    "peraturan": 0.95,
    "official": 0.95,
    "database": 0.85,
    "memo": 0.7,
    "report": 0.7,
    "email": 0.5,
    "catatan": 0.4,
    "legacy": 0.3,      
    "unknown": 0.3,
}


def calculate_recency_score(uploaded_at: str, max_age_days: int = 365) -> float:
    """
    Hitung skor recency (0.0 - 1.0).
    Dokumen hari ini = 1.0, dokumen >max_age_days lalu = 0.1
    """
    if not uploaded_at:
        return 0.1  # Legacy document tanpa tanggal
    
    try:
        if isinstance(uploaded_at, str):
            # Handle ISO format
            doc_date = datetime.fromisoformat(uploaded_at.replace("Z", "+00:00"))
        else:
            doc_date = uploaded_at
        
        now = datetime.now(timezone.utc)
        if doc_date.tzinfo is None:
            doc_date = doc_date.replace(tzinfo=timezone.utc)
        
        age_days = (now - doc_date).days
        # Linear decay: 1.0 → 0.1 over max_age_days
        score = max(0.1, 1.0 - (age_days / max_age_days) * 0.9)
        return round(score, 4)
    except Exception as e:
        logger.debug(f"Recency calculation error: {e}")
        return 0.1


def calculate_version_score(version: str) -> float:
    """
    Hitung skor berdasarkan versi dokumen.
    Versi 1 = 0.1, Versi 2 = 0.5, Versi 3+ = 0.8-1.0
    """
    try:
        v = int(version) if version else 1
    except (ValueError, TypeError):
        v = 1
    
    if v <= 1:
        return 0.1
    elif v == 2:
        return 0.5
    else:
        return min(1.0, 0.5 + (v - 2) * 0.15)


def calculate_source_score(source_type: str) -> float:
    """Ambil skor berdasarkan tipe sumber."""
    return SOURCE_TYPE_SCORES.get(
        (source_type or "unknown").lower(), 
        SOURCE_TYPE_SCORES["unknown"]
    )


def weighted_rerank(results: list, weights: dict = None) -> list:
    """
    Re-rank hasil pencarian dengan bobot gabungan.
    
    Setiap item di results harus punya:
      - score: float (similarity score 0-1)
      - metadata: dict (berisi uploaded_at, version, source_type)
      - text/document: str
    
    Returns: sorted list (skor tertinggi di urutan pertama)
    """
    if not results:
        return results
    
    w = weights or DEFAULT_WEIGHTS
    ranked = []
    
    for item in results:
        meta = item.get("metadata", {})
        
        # Komponen skor
        sim_score = float(item.get("score", 0.5))
        rec_score = calculate_recency_score(meta.get("uploaded_at", ""))
        ver_score = calculate_version_score(meta.get("version", "1"))
        src_score = calculate_source_score(meta.get("source_type", "unknown"))
        
        # Skor gabungan berbobot
        final_score = (
            w["similarity"] * sim_score +
            w["recency"]    * rec_score +
            w["version"]    * ver_score +
            w["source"]     * src_score
        )
        
        item["final_score"] = round(final_score, 4)
        item["score_breakdown"] = {
            "similarity": round(sim_score, 4),
            "recency": round(rec_score, 4),
            "version": round(ver_score, 4),
            "source": round(src_score, 4),
        }
        ranked.append(item)
    
    # Sort descending by final_score
    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked


def filter_superseded(results: list, graph_db=None) -> list:
    """
    Filter out dokumen yang sudah di-supersede oleh dokumen lain.
    
    Cek relasi (newer)-[:SUPERSEDES]->(doc) di Neo4j.
    Jika doc sudah di-supersede DAN doc yang lebih baru juga ada
    di results, maka buang doc lama.
    """
    if not graph_db or not graph_db.driver or not results:
        return results
    
    # Kumpulkan semua ID di result set
    result_ids = {r.get("id", "") for r in results}
    filtered = []
    
    for item in results:
        doc_id = item.get("id", "")
        if not doc_id:
            filtered.append(item)
            continue
        
        # Cek apakah dokumen ini sudah di-supersede
        superseded_by = graph_db.check_superseded_by(doc_id)
        
        if superseded_by:
            # Cek apakah dokumen penggantinya ada di results
            newer_ids = {s.get("newer_id", "") for s in superseded_by}
            if newer_ids & result_ids:
                # Dokumen pengganti ada di results → buang yang lama
                logger.info(
                    f"⚡ Conflict resolved: '{item.get('metadata', {}).get('name', doc_id)}' "
                    f"superseded by newer document"
                )
                item["_superseded"] = True
                item["_superseded_by"] = list(newer_ids & result_ids)
                # Tetap masukkan tapi dengan flag, bisa dipakai untuk logging
                filtered.append(item)
            else:
                # Dokumen pengganti TIDAK ada di results → tetap pakai yang lama
                filtered.append(item)
        else:
            filtered.append(item)
    
    # Pisahkan: non-superseded di depan, superseded di belakang
    active = [r for r in filtered if not r.get("_superseded")]
    obsolete = [r for r in filtered if r.get("_superseded")]
    
    return active + obsolete


def resolve_conflicts(results: list, graph_db=None, weights: dict = None) -> dict:
    """
    Pipeline utama Conflict Resolution.
    
    1. Filter superseded documents (via Neo4j graph)
    2. Weighted re-rank sisanya
    3. Return hasil terurut + conflict report
    
    Returns:
        {
            "results": [...],           # Sorted results
            "conflicts_found": int,      # Jumlah konflik terdeteksi
            "superseded_docs": [...],    # Dokumen yang di-supersede
            "resolution_log": str        # Human-readable log
        }
    """
    if not results:
        return {
            "results": [],
            "conflicts_found": 0,
            "superseded_docs": [],
            "resolution_log": "No results to resolve."
        }
    
    # Step 1: Filter superseded via graph
    filtered = filter_superseded(results, graph_db)
    
    # Step 2: Separate active vs superseded
    active = [r for r in filtered if not r.get("_superseded")]
    superseded = [r for r in filtered if r.get("_superseded")]
    
    # Step 3: Re-rank active results
    ranked = weighted_rerank(active, weights)
    
    # Step 4: Build resolution log
    log_parts = []
    if superseded:
        log_parts.append(f"⚡ {len(superseded)} dokumen usang ditemukan dan diprioritaskan lebih rendah.")
        for doc in superseded:
            name = doc.get("metadata", {}).get("name", doc.get("id", "unknown"))
            log_parts.append(f"  - '{name}' digantikan oleh dokumen yang lebih baru")
    
    if ranked and len(ranked) > 1:
        top = ranked[0]
        top_name = top.get("metadata", {}).get("name", "unknown")
        log_parts.append(
            f"✅ Dokumen terpilih: '{top_name}' "
            f"(skor: {top.get('final_score', 0):.3f})"
        )
    
    return {
        "results": ranked + superseded,  # Active first, then obsolete
        "conflicts_found": len(superseded),
        "superseded_docs": superseded,
        "resolution_log": "\n".join(log_parts) if log_parts else "Tidak ada konflik ditemukan."
    }


# ─── Singleton ───
class ConflictResolver:
    """Wrapper class untuk kemudahan akses dari agents."""
    
    def __init__(self):
        self.weights = DEFAULT_WEIGHTS.copy()
    
    def resolve(self, results: list, graph_db=None) -> dict:
        return resolve_conflicts(results, graph_db, self.weights)
    
    def rerank(self, results: list) -> list:
        return weighted_rerank(results, self.weights)
    
    def set_weights(self, **kwargs):
        """Override bobot tertentu. Contoh: set_weights(recency=0.5)"""
        for k, v in kwargs.items():
            if k in self.weights:
                self.weights[k] = v


conflict_resolver = ConflictResolver()
