"""
retrieval.py — Patent Search Engine
=====================================
Implements the full retrieve-then-rerank pipeline from the proposal.
Called by app.py to serve real search results to the frontend.

Methods:
  r2 — BM25F field-weighted retrieval (baseline)
  r3 — BM25F → Dense re-rank (MaxP chunking)
  r4 — BM25F + Dense weighted fusion (α=0.5)
  r5 — BM25F + Dense Reciprocal Rank Fusion (k=60)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PARSED_PATH   = PROJECT_ROOT / "data" / "interim" / "parsed_patents.jsonl"

# ---------------------------------------------------------------------------
# Settings  (must match notebooks)
# ---------------------------------------------------------------------------
FIELD_WEIGHTS = {"title": 3.0, "abstract": 2.0, "claims": 1.5, "description": 1.0}
FIELDS        = list(FIELD_WEIGHTS.keys())
POOL_SIZE     = 200   # scaled for 20k corpus (was 100 for 2k)
TOKEN_RE      = re.compile(r"[A-Za-z0-9]+")
UCID_RE       = re.compile(r"^[A-Z]{2}-\d+-[A-Z]\d*$", re.IGNORECASE)
TAG_SPACE_RE  = re.compile(r"\s+")

# Dense model — same as Notebook 04
MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 50

# ---------------------------------------------------------------------------
# Module-level state (loaded once on first search)
# ---------------------------------------------------------------------------
_docs_df      = None
_bm25_indexes = None
_dense_model  = None


def _load():
    """Lazy-load corpus + indexes on first call."""
    global _docs_df, _bm25_indexes, _dense_model

    if _docs_df is not None:
        return  # already loaded

    # --- corpus ---
    docs_df = pd.read_json(PARSED_PATH, lines=True)
    docs_df["ucid"] = docs_df["ucid"].astype(str)
    for field in FIELDS + ["ipc"]:
        if field not in docs_df.columns:
            docs_df[field] = ""
        docs_df[field] = docs_df[field].fillna("")

    # --- BM25F indexes ---
    for field in FIELDS:
        docs_df[f"tokens_{field}"] = docs_df[field].map(
            lambda t: TOKEN_RE.findall(t.lower())
        )
    bm25_indexes = {
        field: BM25Okapi(docs_df[f"tokens_{field}"].tolist())
        for field in FIELDS
    }

    # --- Dense model ---
    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    dense_model = SentenceTransformer(MODEL_NAME, device=device)

    _docs_df      = docs_df
    _bm25_indexes = bm25_indexes
    _dense_model  = dense_model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def _search_bm25f(query_text: str, top_k: int = POOL_SIZE) -> pd.DataFrame:
    tokens = _tokenize(query_text)
    scores = np.zeros(len(_docs_df))
    for field, weight in FIELD_WEIGHTS.items():
        scores += weight * _bm25_indexes[field].get_scores(tokens)
    result_df = _docs_df[["ucid", "title"]].copy()
    result_df["bm25f_score"] = scores
    result_df = result_df.sort_values("bm25f_score", ascending=False).head(top_k).reset_index(drop=True)
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    return result_df


def _create_chunks(text: str, title: str):
    if not text or not str(text).strip():
        return []
    words = str(text).split()
    step  = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []
    for i in range(0, len(words), step):
        chunk_text = " ".join(words[i: i + CHUNK_SIZE])
        chunks.append(f"Title: {title} | Text: {chunk_text}")
        if i + CHUNK_SIZE >= len(words):
            break
    return chunks


def _rerank_with_dense(query_text: str, candidates_df: pd.DataFrame) -> dict:
    """Returns {ucid: maxp_score} for each candidate."""
    query_emb   = _dense_model.encode([query_text], normalize_embeddings=True)
    ucids       = candidates_df["ucid"].tolist()
    all_chunks  = []
    chunk_ucids = []

    for ucid in ucids:
        row = _docs_df[_docs_df["ucid"] == ucid]
        if row.empty:
            continue
        r          = row.iloc[0]
        title_text = (r["title"] or "Unknown Title").strip()
        full_text  = TAG_SPACE_RE.sub(
            " ", f"{r['abstract']} {r['description']} {r['claims']}"
        ).strip()
        chunks = _create_chunks(full_text, title_text)
        if not chunks:
            chunks = [f"Title: {title_text} | Text: "]
        all_chunks.extend(chunks)
        chunk_ucids.extend([ucid] * len(chunks))

    chunk_embs  = _dense_model.encode(all_chunks, normalize_embeddings=True, show_progress_bar=False)
    chunk_scores = (query_emb @ chunk_embs.T).flatten()

    patent_scores = {}
    for ucid, score in zip(chunk_ucids, chunk_scores):
        if ucid not in patent_scores or score > patent_scores[ucid]:
            patent_scores[ucid] = float(score)
    return patent_scores


def _normalise_to_100(scores: list[float]) -> list[float]:
    """Min-max scale a list of scores to [0, 100] for display."""
    lo, hi = min(scores), max(scores)
    rng = hi - lo if hi > lo else 1.0
    return [round((s - lo) / rng * 100, 1) for s in scores]


def _snippet(text: str, max_words: int = 35) -> str:
    if not text:
        return ""
    words = text.split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")


def _build_result(ucid: str, display_score: float) -> dict:
    row = _docs_df[_docs_df["ucid"] == ucid]
    if row.empty:
        return {"ucid": ucid, "title": "", "ipc": "", "score": display_score, "snippet": ""}
    r = row.iloc[0]
    return {
        "ucid":    ucid,
        "title":   r["title"],
        "ipc":     r["ipc"] or "—",
        "score":   display_score,
        "snippet": _snippet(r["abstract"]),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_patent(ucid: str) -> dict | None:
    """Return full patent data for a given UCID, or None if not found."""
    _load()
    row = _docs_df[_docs_df["ucid"] == ucid]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "ucid":        ucid,
        "title":       r["title"]       or "",
        "ipc":         r["ipc"]         or "—",
        "abstract":    r["abstract"]    or "",
        "claims":      r["claims"]      or "",
        "description": r["description"] or "",
    }


def search(query: str, method: str = "r2", top_k: int = 10) -> list[dict]:
    """
    Run patent search using the chosen method.

    Args:
        query:   Free-text query string.
        method:  'r2' | 'r3' | 'r4' | 'r5'
        top_k:   Number of results to return.

    Returns:
        List of dicts with keys: ucid, title, ipc, score (0–100), snippet.
    """
    _load()

    # Direct UCID lookup — bypass BM25F entirely
    if UCID_RE.match(query.strip()):
        ucid = query.strip().upper()
        row  = _docs_df[_docs_df["ucid"] == ucid]
        if not row.empty:
            return [_build_result(ucid, 100.0)]
        return []   # UCID not found in corpus

    candidates = _search_bm25f(query, top_k=POOL_SIZE)

    if method == "r2":
        # BM25F only
        top = candidates.head(top_k)
        raw_scores = top["bm25f_score"].tolist()
        norm = _normalise_to_100(raw_scores)
        return [_build_result(row["ucid"], norm[i])
                for i, (_, row) in enumerate(top.iterrows())]

    # All hybrid methods need dense scores
    dense_scores = _rerank_with_dense(query, candidates)
    bm25f_scores = dict(zip(candidates["ucid"], candidates["bm25f_score"]))

    if method == "r3":
        ranked = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        raw    = [s for _, s in ranked]
        norm   = _normalise_to_100(raw)
        return [_build_result(ucid, norm[i]) for i, (ucid, _) in enumerate(ranked)]

    def minmax(d):
        vals = list(d.values())
        lo, hi = min(vals), max(vals)
        rng = hi - lo if hi > lo else 1.0
        return {k: (v - lo) / rng for k, v in d.items()}

    bm25f_norm = minmax(bm25f_scores)
    dense_norm = minmax(dense_scores)

    if method == "r4":
        alpha    = 0.5
        combined = {
            ucid: alpha * bm25f_norm[ucid] + (1 - alpha) * dense_norm[ucid]
            for ucid in bm25f_norm
        }
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        raw    = [s for _, s in ranked]
        norm   = _normalise_to_100(raw)
        return [_build_result(ucid, norm[i]) for i, (ucid, _) in enumerate(ranked)]

    if method == "r5":
        k          = 60
        bm25f_ranks = dict(zip(candidates["ucid"], candidates["rank"]))
        dense_ranked = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
        dense_ranks  = {ucid: i + 1 for i, (ucid, _) in enumerate(dense_ranked)}
        rrf = {
            ucid: 1.0 / (k + bm25f_ranks[ucid]) + 1.0 / (k + dense_ranks[ucid])
            for ucid in bm25f_ranks
        }
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        raw    = [s for _, s in ranked]
        norm   = _normalise_to_100(raw)
        return [_build_result(ucid, norm[i]) for i, (ucid, _) in enumerate(ranked)]

    raise ValueError(f"Unknown method '{method}'. Choose r2, r3, r4 or r5.")
