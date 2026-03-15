# app.py
"""
Reconciliation service (hybrid string + local embeddings) with safe embedding handling.
- Uses local sentence-transformers embeddings when USE_LOCAL_EMBEDDINGS=true
- Annoy approximate nearest neighbor index for semantic search
- Robust string heuristics (Jaro-Winkler, RapidFuzz, token overlap)
- Safe embedding persistence (avoid saving all-zero embeddings)
- Admin-protected data extension endpoints (X-ADMIN-TOKEN or admin_token)
- Environment flags: USE_LOCAL_EMBEDDINGS, SKIP_EMBEDDINGS_ON_START, BUILD_BATCH_SIZE, APP_DEBUG
"""

import os
import csv
import json
import logging
import unicodedata
import re
import threading
import time
import random
from typing import List, Dict, Tuple, Optional

from flask import Flask, request, jsonify, abort
from flask_cors import CORS

# Similarity utilities
from difflib import SequenceMatcher
try:
    import jellyfish
    _HAS_JELLYFISH = True
except Exception:
    _HAS_JELLYFISH = False

try:
    from rapidfuzz import fuzz as rf_fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# Numeric and index libraries
import numpy as np
try:
    from annoy import AnnoyIndex
    _HAS_ANNOY = True
except Exception:
    _HAS_ANNOY = False

# Local embedding model (sentence-transformers)
_USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() in ("1", "true", "yes")
_LOCAL_MODEL = None
if _USE_LOCAL:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        SentenceTransformer = None

# Caching utilities
from cachetools import TTLCache, cached

# dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- Configuration ----------------
CSV_PATH = os.getenv("PLACES_CSV", "places.csv")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "embeddings.npz")
ANN_INDEX_PATH = os.getenv("ANN_INDEX_PATH", "ann_index.ann")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", None)
HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = int(os.getenv("APP_PORT", 5000))
DEBUG = os.getenv("APP_DEBUG", "false").lower() in ("1", "true", "yes")

# Scoring weights and thresholds
STRING_WEIGHT = float(os.getenv("STRING_WEIGHT", 0.6))
EMBED_WEIGHT = float(os.getenv("EMBED_WEIGHT", 0.4))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", 0.95))

# Annoy parameters
ANNOY_METRIC = os.getenv("ANNOY_METRIC", "angular")
ANNOY_TREES = int(os.getenv("ANNOY_TREES", 10))

# Embedding dimension default (all-MiniLM-L6-v2 -> 384)
EMBED_DIM = int(os.getenv("EMBED_DIM", 384))

# Control flags
SKIP_EMBEDDINGS_ON_START = os.getenv("SKIP_EMBEDDINGS_ON_START", "false").lower() in ("1", "true", "yes")
BUILD_BATCH_SIZE = int(os.getenv("BUILD_BATCH_SIZE", 8))

# Caches
QUERY_EMBED_CACHE = TTLCache(maxsize=1024, ttl=60 * 60)
RESULT_CACHE = TTLCache(maxsize=1024, ttl=60 * 5)

# Logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)
# reduce noisy debug logs from lower-level libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Flask app
app = Flask(__name__)
CORS(app)

# In-memory structures
ITEMS: List[Dict] = []
_PREFIX_INDEX: Dict[str, List[Dict]] = {}
_ID_TO_INDEX: Dict[str, int] = {}
_EMBEDDINGS: Optional[np.ndarray] = None
_ANN_INDEX: Optional[AnnoyIndex] = None
_ANN_LOCK = threading.Lock()

# ---------------- Exceptions ----------------

class EmbeddingError(Exception):
    """Generic embedding error."""
    pass

# ---------------- Text utilities ----------------

def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s\-\.\,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str) -> List[str]:
    return [t for t in normalize_text(s).split(" ") if t]

def jaccard(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    inter = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(inter) / len(union)

def seq_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def jaro_winkler(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _HAS_JELLYFISH:
        try:
            return jellyfish.jaro_winkler_similarity(a, b)
        except Exception:
            return seq_ratio(a, b)
    return seq_ratio(a, b)

def rapidfuzz_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        try:
            return rf_fuzz.token_sort_ratio(a, b) / 100.0
        except Exception:
            return seq_ratio(a, b)
    return seq_ratio(a, b)

def combined_name_score(query: str, candidate: str) -> float:
    nq = normalize_text(query)
    nc = normalize_text(candidate)
    if not nq or not nc:
        return 0.0
    if nq == nc:
        return 1.0
    t_q, t_c = tokens(nq), tokens(nc)
    j = jaccard(t_q, t_c)
    s = seq_ratio(nq, nc)
    jw = jaro_winkler(nq, nc)
    rf = rapidfuzz_ratio(nq, nc)
    weight_j = 0.50 if (len(t_q) > 1 or len(t_c) > 1) else 0.30
    weight_s = 0.15
    weight_jw = 0.15
    weight_rf = max(0.0, 1.0 - (weight_j + weight_s + weight_jw))
    base = weight_j * j + weight_s * s + weight_jw * jw + weight_rf * rf
    if nq in nc or nc in nq:
        return max(base, 0.85)
    return max(0.0, min(1.0, base))

# ---------------- CSV loading and prefix index ----------------

def load_items_from_csv(path: str = CSV_PATH) -> List[Dict]:
    items = []
    if not os.path.exists(path):
        logger.warning("CSV not found: %s", path)
        return items
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            alt_raw = row.get("aliases") or row.get("alt_names") or ""
            alt_list = [a.strip() for a in alt_raw.split(";") if a.strip()]
            items.append({
                "id": (row.get("id") or "").strip(),
                "name": (row.get("name") or "").strip(),
                "country": (row.get("country") or "").strip(),
                "region": (row.get("region") or "").strip(),
                "type": (row.get("type") or "").strip(),
                "aliases": alt_list,
                "description": (row.get("description") or "").strip()
            })
    logger.info("Loaded %d items from %s", len(items), path)
    return items

def build_prefix_index(items: List[Dict], max_prefix_len: int = 6):
    global _PREFIX_INDEX
    _PREFIX_INDEX = {}
    for it in items:
        candidates = [it.get("name", "")] + it.get("aliases", [])
        for cand in candidates:
            n = normalize_text(cand)
            for L in range(1, min(len(n), max_prefix_len) + 1):
                pref = n[:L]
                _PREFIX_INDEX.setdefault(pref, []).append(it)
    logger.debug("Built prefix index with %d keys", len(_PREFIX_INDEX))

# ---------------- Local embedder helpers ----------------

def get_local_embedder():
    """Load and return the sentence-transformers model (singleton)."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        if 'SentenceTransformer' not in globals() or SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed; install sentence-transformers and torch")
        _LOCAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _LOCAL_MODEL

def _embeddings_nonzero(emb: np.ndarray) -> bool:
    if emb is None:
        return False
    return bool(np.any(np.abs(emb) > 1e-6))

def safe_save_embeddings(ids: List[str], embeddings: np.ndarray, path: str = EMBEDDINGS_PATH):
    if embeddings is None or embeddings.size == 0:
        logger.warning("No embeddings to save.")
        return
    if not _embeddings_nonzero(embeddings):
        logger.warning("Embeddings appear to be all zeros; not saving to disk.")
        return
    np.savez_compressed(path, ids=np.array(ids), embeddings=embeddings)
    logger.info("Saved embeddings to %s", path)

# ---------------- Embedding computation (local only) ----------------

def compute_embeddings_for_texts(texts: List[str], batch_size: int = BUILD_BATCH_SIZE) -> np.ndarray:
    """
    Compute embeddings using local model (sentence-transformers).
    Returns numpy array (n, dim).
    """
    global EMBED_DIM
    if not _USE_LOCAL:
        raise EmbeddingError("Local embeddings disabled. Set USE_LOCAL_EMBEDDINGS=true.")
    model = get_local_embedder()
    all_embs = []
    # chunk to control memory
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embs.append(np.array(embs, dtype=np.float32))
    arr = np.vstack(all_embs) if all_embs else np.zeros((0, EMBED_DIM), dtype=np.float32)
    if arr.size:
        EMBED_DIM = arr.shape[1]
    return arr

def load_embeddings(path: str = EMBEDDINGS_PATH) -> Tuple[List[str], Optional[np.ndarray]]:
    if not os.path.exists(path):
        return [], None
    data = np.load(path, allow_pickle=True)
    ids = data["ids"].tolist()
    embeddings = data["embeddings"]
    logger.info("Loaded embeddings from %s (n=%d, dim=%d)", path, len(ids), embeddings.shape[1])
    return ids, embeddings

def build_annoy_index(embeddings: np.ndarray, dim: int, path: str = ANN_INDEX_PATH, trees: int = ANNOY_TREES) -> AnnoyIndex:
    if not _HAS_ANNOY:
        raise RuntimeError("annoy not installed")
    idx = AnnoyIndex(dim, metric=ANNOY_METRIC)
    for i in range(embeddings.shape[0]):
        idx.add_item(i, embeddings[i].astype(np.float32))
    idx.build(trees)
    idx.save(path)
    logger.info("Built and saved Annoy index to %s", path)
    return idx

def load_annoy_index(dim: int, path: str = ANN_INDEX_PATH) -> Optional[AnnoyIndex]:
    if not _HAS_ANNOY:
        return None
    if not os.path.exists(path):
        return None
    idx = AnnoyIndex(dim, metric=ANNOY_METRIC)
    idx.load(path)
    logger.info("Loaded Annoy index from %s", path)
    return idx

# ---------------- Initialization ----------------

def initialize_all():
    global ITEMS, _EMBEDDINGS, _ANN_INDEX, _ID_TO_INDEX, EMBED_DIM
    ITEMS = load_items_from_csv(CSV_PATH)
    build_prefix_index(ITEMS)
    if SKIP_EMBEDDINGS_ON_START:
        logger.info("SKIP_EMBEDDINGS_ON_START is true; skipping embedding computation on startup.")
        _EMBEDDINGS = None
    else:
        ids, embeddings = load_embeddings(EMBEDDINGS_PATH)
        if embeddings is None or len(ids) != len(ITEMS):
            texts = []
            ids = []
            for it in ITEMS:
                parts = [it.get("name", "")]
                parts += it.get("aliases", [])
                parts += [it.get("country", ""), it.get("region", ""), it.get("type", ""), it.get("description", "")]
                texts.append(" | ".join([p for p in parts if p]))
                ids.append(it.get("id"))
            if texts and _USE_LOCAL:
                try:
                    emb = compute_embeddings_for_texts(texts)
                except Exception as e:
                    logger.exception("Failed to compute local embeddings: %s", e)
                    _EMBEDDINGS = None
                else:
                    _EMBEDDINGS = emb
                    EMBED_DIM = emb.shape[1]
                    safe_save_embeddings(ids, emb)
            else:
                _EMBEDDINGS = None
        else:
            _EMBEDDINGS = embeddings
            EMBED_DIM = _EMBEDDINGS.shape[1]
    if _EMBEDDINGS is not None and _EMBEDDINGS.shape[0] > 0:
        try:
            idx = load_annoy_index(_EMBEDDINGS.shape[1])
            if idx is None:
                idx = build_annoy_index(_EMBEDDINGS, _EMBEDDINGS.shape[1])
            _ANN_INDEX = idx
        except Exception as e:
            logger.exception("Failed to build/load Annoy index: %s", e)
            _ANN_INDEX = None
    else:
        _ANN_INDEX = None
    _ID_TO_INDEX = {}
    if _EMBEDDINGS is not None:
        for i, it in enumerate(ITEMS):
            _ID_TO_INDEX[it.get("id")] = i
    logger.info("Initialization complete: items=%d embeddings=%s annoy=%s", len(ITEMS),
                "yes" if _EMBEDDINGS is not None else "no",
                "yes" if _ANN_INDEX is not None else "no")

# ---------------- Embedding helpers (query) ----------------

@cached(QUERY_EMBED_CACHE)
def get_query_embedding(text: str) -> np.ndarray:
    """Return embedding for a query string (cached). Uses local model if configured."""
    use_local = _USE_LOCAL
    if use_local:
        model = get_local_embedder()
        emb = model.encode([text], convert_to_numpy=True)[0]
        return np.array(emb, dtype=np.float32)
    # deterministic fallback vector (weak) to allow string-only behavior
    return np.array([float(sum(ord(c) for c in text) % 1000)] * EMBED_DIM, dtype=np.float32)

# ---------------- Matching and scoring ----------------

def best_name_match_score(query: str, item: Dict) -> Tuple[float, str]:
    best = 0.0
    best_str = item.get("name", "")
    s = combined_name_score(query, item.get("name", ""))
    if s > best:
        best = s; best_str = item.get("name", "")
    for alt in item.get("aliases", []):
        s2 = combined_name_score(query, alt)
        if s2 > best:
            best = s2; best_str = alt
    return best, best_str

def field_match_boost(query_tokens: List[str], field_value: str) -> float:
    if not field_value:
        return 0.0
    ftoks = tokens(field_value)
    if not ftoks:
        return 0.0
    inter = set(query_tokens).intersection(set(ftoks))
    if not inter:
        return 0.0
    return len(inter) / len(ftoks)

def string_item_score(query: str, item: Dict, weights: Dict = None) -> Tuple[float, Dict]:
    if weights is None:
        weights = {"name": 0.65, "country": 0.15, "region": 0.12, "type": 0.08}
    nq = normalize_text(query)
    q_toks = tokens(nq)
    name_score, matched_str = best_name_match_score(query, item)
    country_boost = field_match_boost(q_toks, item.get("country", ""))
    region_boost = field_match_boost(q_toks, item.get("region", ""))
    type_boost = field_match_boost(q_toks, item.get("type", ""))
    if item.get("country") and normalize_text(item.get("country")) in nq:
        country_boost = max(country_boost, 1.0)
    if item.get("region") and normalize_text(item.get("region")) in nq:
        region_boost = max(region_boost, 1.0)
    if item.get("type") and normalize_text(item.get("type")) in nq:
        type_boost = max(type_boost, 1.0)
    score = (weights["name"] * name_score +
             weights["country"] * country_boost +
             weights["region"] * region_boost +
             weights["type"] * type_boost)
    score = max(0.0, min(1.0, score))
    details = {
        "name_score": round(name_score, 4),
        "country_boost": round(country_boost, 4),
        "region_boost": round(region_boost, 4),
        "type_boost": round(type_boost, 4),
        "matched_string": matched_str
    }
    return score, details

def embed_candidates(query_emb: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
    global _ANN_INDEX, _EMBEDDINGS
    if _ANN_INDEX is None or _EMBEDDINGS is None or _EMBEDDINGS.shape[0] == 0:
        return []
    with _ANN_LOCK:
        try:
            idxs, dists = _ANN_INDEX.get_nns_by_vector(query_emb.astype(np.float32), top_k, include_distances=True)
            sims = []
            for dist in dists:
                if ANNOY_METRIC == "angular":
                    cos_sim = max(-1.0, min(1.0, 1.0 - (dist ** 2) / 2.0))
                else:
                    cos_sim = 1.0 - dist
                sims.append(float(cos_sim))
            return list(zip(idxs, sims))
        except Exception as e:
            logger.exception("Annoy query failed: %s", e)
            return []

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def hybrid_score_for_item(query: str, query_emb: np.ndarray, item: Dict, item_index: Optional[int]) -> Tuple[float, Dict]:
    string_score, s_details = string_item_score(query, item)
    embed_score = 0.0
    if query_emb is not None and item_index is not None and _EMBEDDINGS is not None:
        try:
            item_emb = _EMBEDDINGS[item_index]
            embed_score = cosine_similarity(query_emb, item_emb)
            embed_score = max(0.0, (embed_score + 1.0) / 2.0)
        except Exception:
            embed_score = 0.0
    final = STRING_WEIGHT * string_score + EMBED_WEIGHT * embed_score
    details = {
        "string_score": round(string_score, 4),
        "embed_score": round(embed_score, 4),
        **s_details
    }
    return final, details

# ---------------- Find matches and suggest ----------------

def find_matches(query: str, limit: int = 5, cutoff: float = 0.55) -> List[Tuple[float, Dict, Dict]]:
    if not query:
        return []
    nq = normalize_text(query)
    q_toks = tokens(nq)
    candidates = set()
    for L in range(len(nq), 0, -1):
        key = nq[:L]
        if key in _PREFIX_INDEX:
            for it in _PREFIX_INDEX[key]:
                candidates.add(it.get("id"))
            break
    for it in ITEMS:
        if it.get("country") and normalize_text(it.get("country")) in nq:
            candidates.add(it.get("id"))
        if it.get("region") and normalize_text(it.get("region")) in nq:
            candidates.add(it.get("id"))
    if not candidates:
        candidates = {it.get("id") for it in ITEMS}
    query_emb = get_query_embedding(query)
    embed_neigh = embed_candidates(query_emb, top_k=limit * 3)
    embed_candidate_ids = {ITEMS[idx].get("id") for idx, _ in embed_neigh if 0 <= idx < len(ITEMS)}
    candidates = candidates.union(embed_candidate_ids)
    scored = []
    for cid in candidates:
        idx = _ID_TO_INDEX.get(cid)
        item = next((i for i in ITEMS if i.get("id") == cid), None)
        if item is None:
            continue
        score, details = hybrid_score_for_item(query, query_emb, item, idx)
        if score >= cutoff:
            scored.append((score, item, details))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]

def suggest_entities(prefix: str, limit: int = 10) -> List[Dict]:
    if not prefix:
        return []
    npref = normalize_text(prefix)
    ptoks = tokens(npref)
    suggestions = []
    candidates = []
    for L in range(len(npref), 0, -1):
        key = npref[:L]
        if key in _PREFIX_INDEX:
            candidates.extend(_PREFIX_INDEX[key])
            break
    if not candidates:
        candidates = ITEMS
    seen = set()
    for it in candidates:
        if it.get("id") in seen:
            continue
        seen.add(it.get("id"))
        name_norm = normalize_text(it.get("name", ""))
        score = 0.0
        if name_norm.startswith(npref):
            score = 0.95
        else:
            name_toks = tokens(name_norm)
            j = jaccard(ptoks, name_toks)
            if j > 0:
                score = 0.6 + 0.4 * j
            else:
                for alt in it.get("aliases", []):
                    alt_norm = normalize_text(alt)
                    if alt_norm.startswith(npref):
                        score = max(score, 0.7)
                    else:
                        alt_toks = tokens(alt_norm)
                        j2 = jaccard(ptoks, alt_toks)
                        if j2 > 0:
                            score = max(score, 0.5 + 0.4 * j2)
        if score > 0:
            suggestions.append((score, it))
    suggestions.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, it in suggestions[:limit]:
        out.append({
            "id": it.get("id"),
            "name": it.get("name"),
            "score": int(round(s * 100)),
            "type": [{"id": "Place", "name": "Place"}]
        })
    return out

# ---------------- API endpoints ----------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "items": len(ITEMS)})

@app.route("/service", methods=["GET", "POST"])
def service_metadata():
    return jsonify({
        "name": "Places Reconciliation (Hybrid)",
        "identifierSpace": "http://example.org/places",
        "schemaSpace": "http://example.org/schema",
        "view": {"url": f"http://{HOST}:{PORT}/view/{{{{id}}}}"},
        "defaultTypes": [{"id": "Place", "name": "Place"}],
        "preview": {"url": f"http://{HOST}:{PORT}/view/{{{{id}}}}", "width": 400, "height": 200},
        "properties": [{"id": "country", "name": "Country"}, {"id": "region", "name": "Region"}, {"id": "type", "name": "Type"}]
    })

@app.route("/view/<entity_id>", methods=["GET"])
def view_entity(entity_id):
    item = next((i for i in ITEMS if i.get("id") == entity_id), None)
    if not item:
        return "Not found", 404
    html = f"<h1>{item.get('name')}</h1>"
    if item.get("type"):
        html += f"<p><strong>Type:</strong> {item.get('type')}</p>"
    if item.get("country"):
        html += f"<p><strong>Country:</strong> {item.get('country')}</p>"
    if item.get("region"):
        html += f"<p><strong>Region:</strong> {item.get('region')}</p>"
    if item.get("description"):
        html += f"<p>{item.get('description')}</p>"
    if item.get("aliases"):
        html += "<p><strong>Aliases:</strong> " + ", ".join(item["aliases"]) + "</p>"
    return html

@app.route("/reconcile", methods=["GET", "POST"])
def reconcile():
    payload = request.get_json(silent=True)
    if payload is None:
        if "queries" in request.form:
            try:
                payload = {"queries": json.loads(request.form["queries"])}
            except Exception:
                payload = None
        else:
            raw = request.get_data(as_text=True)
            if raw:
                try:
                    payload = json.loads(raw)
                except Exception:
                    payload = None
    if isinstance(payload, dict) and "queries" in payload:
        out = {}
        queries_obj = payload["queries"]
        if isinstance(queries_obj, str):
            try:
                queries_obj = json.loads(queries_obj)
            except Exception:
                queries_obj = {}
        for key, qobj in queries_obj.items():
            qtext = qobj.get("query", "") if isinstance(qobj, dict) else ""
            limit = int(qobj.get("limit", 5)) if isinstance(qobj, dict) and "limit" in qobj else 5
            cutoff = float(qobj.get("cutoff", 0.55)) if isinstance(qobj, dict) and "cutoff" in qobj else 0.55
            matches = find_matches(qtext, limit=limit, cutoff=cutoff)
            out[key] = {"result": [
                {
                    "id": m[1].get("id"),
                    "name": m[1].get("name"),
                    "score": int(round(m[0] * 100)),
                    "match": m[0] >= MATCH_THRESHOLD,
                    "type": [{"id": "Place", "name": "Place"}],
                    "description": m[1].get("description", ""),
                    "metadata": {"country": m[1].get("country", ""), "region": m[1].get("region", ""), "type": m[1].get("type", "")}
                } for m in matches
            ]}
        return jsonify(out)
    if isinstance(payload, dict) and "query" in payload:
        qtext = payload.get("query", "")
        limit = int(payload.get("limit", 5))
        cutoff = float(payload.get("cutoff", 0.55))
    else:
        qtext = request.args.get("query", "")
        limit = int(request.args.get("limit", 5))
        cutoff = float(request.args.get("cutoff", 0.55))
    matches = find_matches(qtext, limit=limit, cutoff=cutoff)
    return jsonify({"result": [
        {
            "id": m[1].get("id"),
            "name": m[1].get("name"),
            "score": int(round(m[0] * 100)),
            "match": m[0] >= MATCH_THRESHOLD,
            "type": [{"id": "Place", "name": "Place"}],
            "description": m[1].get("description", ""),
            "metadata": {"country": m[1].get("country", ""), "region": m[1].get("region", ""), "type": m[1].get("type", "")},
            "details": m[2]
        } for m in matches
    ]})

@app.route("/suggest", methods=["GET"])
def suggest():
    q = request.args.get("q", "")
    limit = int(request.args.get("limit", 10))
    suggestions = suggest_entities(q, limit=limit)
    return jsonify({"result": suggestions})

# ---------------- Data extension endpoints (protected) ----------------

def require_admin():
    token = request.headers.get("X-ADMIN-TOKEN") or request.args.get("admin_token")
    if not ADMIN_TOKEN:
        abort(403, description="Admin token not configured on server")
    if not token or token != ADMIN_TOKEN:
        abort(401, description="Unauthorized")

def persist_items_to_csv(path: str = CSV_PATH):
    tmp = path + ".tmp"
    fieldnames = ["id", "name", "country", "region", "type", "aliases", "description"]
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for it in ITEMS:
            writer.writerow({
                "id": it.get("id", ""),
                "name": it.get("name", ""),
                "country": it.get("country", ""),
                "region": it.get("region", ""),
                "type": it.get("type", ""),
                "aliases": ";".join(it.get("aliases", [])),
                "description": it.get("description", "")
            })
    os.replace(tmp, path)
    logger.info("Persisted %d items to %s", len(ITEMS), path)

def rebuild_embeddings_and_index():
    global _EMBEDDINGS, _ANN_INDEX, _ID_TO_INDEX, EMBED_DIM
    texts = []
    ids = []
    for it in ITEMS:
        parts = [it.get("name", "")]
        parts += it.get("aliases", [])
        parts += [it.get("country", ""), it.get("region", ""), it.get("type", ""), it.get("description", "")]
        texts.append(" | ".join([p for p in parts if p]))
        ids.append(it.get("id"))
    if texts and _USE_LOCAL:
        try:
            emb = compute_embeddings_for_texts(texts)
        except Exception as e:
            logger.exception("Embedding rebuild failed: %s", e)
            _EMBEDDINGS = None
        else:
            _EMBEDDINGS = emb
            EMBED_DIM = emb.shape[1]
            safe_save_embeddings(ids, emb)
            if _HAS_ANNOY:
                with _ANN_LOCK:
                    _ANN_INDEX = build_annoy_index(_EMBEDDINGS, _EMBEDDINGS.shape[1])
    else:
        _EMBEDDINGS = None
    _ID_TO_INDEX = {it.get("id"): i for i, it in enumerate(ITEMS)}
    build_prefix_index(ITEMS)

@app.route("/items", methods=["POST"])
def add_item():
    require_admin()
    payload = request.get_json(silent=True)
    if not payload:
        abort(400, description="Invalid JSON")
    new = {
        "id": str(payload.get("id") or ""),
        "name": str(payload.get("name") or ""),
        "country": str(payload.get("country") or ""),
        "region": str(payload.get("region") or ""),
        "type": str(payload.get("type") or ""),
        "aliases": payload.get("aliases") if isinstance(payload.get("aliases"), list) else [a.strip() for a in (payload.get("aliases") or "").split(";") if a.strip()],
        "description": str(payload.get("description") or "")
    }
    if not new["id"] or not new["name"]:
        abort(400, description="id and name required")
    ITEMS.append(new)
    persist_items_to_csv(CSV_PATH)
    rebuild_embeddings_and_index()
    return jsonify({"status": "added", "id": new["id"]})

@app.route("/items/<entity_id>", methods=["PUT"])
def update_item(entity_id):
    require_admin()
    payload = request.get_json(silent=True)
    if not payload:
        abort(400, description="Invalid JSON")
    item = next((i for i in ITEMS if i.get("id") == entity_id), None)
    if not item:
        abort(404, description="Not found")
    for k in ("name", "country", "region", "type", "description"):
        if k in payload:
            item[k] = str(payload[k] or "")
    if "aliases" in payload:
        item["aliases"] = payload.get("aliases") if isinstance(payload.get("aliases"), list) else [a.strip() for a in (payload.get("aliases") or "").split(";") if a.strip()]
    persist_items_to_csv(CSV_PATH)
    rebuild_embeddings_and_index()
    return jsonify({"status": "updated", "id": entity_id})

@app.route("/items/<entity_id>", methods=["DELETE"])
def delete_item(entity_id):
    require_admin()
    global ITEMS
    before = len(ITEMS)
    ITEMS = [i for i in ITEMS if i.get("id") != entity_id]
    after = len(ITEMS)
    if before == after:
        abort(404, description="Not found")
    persist_items_to_csv(CSV_PATH)
    rebuild_embeddings_and_index()
    return jsonify({"status": "deleted", "id": entity_id})

@app.route("/_reload_items", methods=["POST"])
def _reload_items():
    global ITEMS
    ITEMS = load_items_from_csv(CSV_PATH)
    build_prefix_index(ITEMS)
    t = threading.Thread(target=rebuild_embeddings_and_index, daemon=True)
    t.start()
    return jsonify({"status": "reloading", "count": len(ITEMS)})

# ---------------- Main ----------------

if __name__ == "__main__":
    if not _HAS_ANNOY:
        logger.warning("Annoy not available; embedding-based search will be disabled")
    initialize_all()
    app.run(host=HOST, port=PORT, debug=DEBUG)

