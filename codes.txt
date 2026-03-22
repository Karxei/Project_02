# app.py
import os
import csv
import json
import logging
import unicodedata
import re
import threading
from typing import List, Dict, Tuple, Optional

from flask import Flask, request, jsonify, abort
from flask_cors import CORS

from difflib import SequenceMatcher
import numpy as np

# Optional similarity libraries
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

# Optional Annoy index
try:
    from annoy import AnnoyIndex
    _HAS_ANNOY = True
except Exception:
    _HAS_ANNOY = False

# Local embeddings
_USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() in ("1", "true", "yes")
_LOCAL_MODEL = None
if _USE_LOCAL:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        SentenceTransformer = None

from cachetools import TTLCache, cached

# Optional dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- Config ----------------

CSV_PATH = os.getenv("PLACES_CSV", "places.csv")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "embeddings.npz")
ANN_INDEX_PATH = os.getenv("ANN_INDEX_PATH", "ann_index.ann")

# Strip whitespace so trailing spaces in .env don't break equality
_ADMIN_ENV = os.getenv("ADMIN_TOKEN", "")
ADMIN_TOKEN = _ADMIN_ENV.strip() if _ADMIN_ENV else None

HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = int(os.getenv("APP_PORT", 5000))
DEBUG = os.getenv("APP_DEBUG", "false").lower() in ("1", "true", "yes")

STRING_WEIGHT = float(os.getenv("STRING_WEIGHT", 0.8))
EMBED_WEIGHT = float(os.getenv("EMBED_WEIGHT", 0.2))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", 0.55))
LLM_WEIGHT = float(os.getenv("LLM_WEIGHT", 0.7))

ANNOY_METRIC = os.getenv("ANNOY_METRIC", "angular")
ANNOY_TREES = int(os.getenv("ANNOY_TREES", 10))
EMBED_DIM = int(os.getenv("EMBED_DIM", 384))

SKIP_EMBEDDINGS_ON_START = os.getenv("SKIP_EMBEDDINGS_ON_START", "false").lower() in ("1", "true", "yes")
BUILD_BATCH_SIZE = int(os.getenv("BUILD_BATCH_SIZE", 8))

QUERY_EMBED_CACHE = TTLCache(maxsize=1024, ttl=3600)

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

if _USE_LOCAL and (SentenceTransformer is None):
    logger.warning("USE_LOCAL_EMBEDDINGS=true but sentence-transformers is not installed; falling back to pseudo-embeddings.")
if not _HAS_ANNOY:
    logger.info("Annoy is not installed; embedding ANN index will be disabled.")

# Flask app
app = Flask(__name__)
CORS(app)

# In-memory state
ITEMS: List[Dict] = []
_PREFIX_INDEX: Dict[str, List[Dict]] = {}
_ID_TO_INDEX: Dict[str, int] = {}
_EMBEDDINGS: Optional[np.ndarray] = None
_ANN_INDEX: Optional["AnnoyIndex"] = None
_ANN_LOCK = threading.Lock()

# Blocking indices
_COUNTRY_INDEX: Dict[str, List[str]] = {}
_REGION_INDEX: Dict[str, List[str]] = {}
_TYPE_INDEX: Dict[str, List[str]] = {}
_TOKEN_PREFIX_INDEX: Dict[str, List[str]] = {}

# Print admin token for dev mode only
if DEBUG or os.getenv("SHOW_ADMIN_TOKEN", "false").lower() in ("1", "true", "yes"):
    print("ADMIN_TOKEN LOADED:", repr(ADMIN_TOKEN))


# ---------------- Text utilities ----------------

def normalize_text(s: Optional[str]) -> str:
    """Lowercase, strip accents, remove punctuation, collapse spaces."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s\-\.\,']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens(s: str) -> List[str]:
    """Tokenize normalized text into simple whitespace tokens."""
    return [t for t in normalize_text(s).split(" ") if t]


def jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard similarity between two token lists."""
    if not a and not b:
        return 0.0
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def seq_ratio(a: str, b: str) -> float:
    """SequenceMatcher ratio as a fallback similarity."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler similarity if available, else SequenceMatcher."""
    if not a or not b:
        return 0.0
    if _HAS_JELLYFISH:
        try:
            return jellyfish.jaro_winkler_similarity(a, b)
        except Exception:
            return seq_ratio(a, b)
    return seq_ratio(a, b)


def rapidfuzz_ratio(a: str, b: str) -> float:
    """RapidFuzz token_sort_ratio if available, else SequenceMatcher."""
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        try:
            return rf_fuzz.token_sort_ratio(a, b) / 100.0
        except Exception:
            return seq_ratio(a, b)
    return seq_ratio(a, b)


def is_capital(item: Dict) -> bool:
    """Heuristic: description contains 'capital'."""
    return "capital" in normalize_text(item.get("description", ""))


def combined_name_score(query: str, candidate: str) -> float:
    """
    Combine multiple string similarities into a single name score.
    Includes substring boost for abbreviations/short forms.
    """
    nq = normalize_text(query)
    nc = normalize_text(candidate)
    if not nq or not nc:
        return 0.0
    if nq == nc:
        return 1.0
    tq, tc = tokens(nq), tokens(nc)
    j = jaccard(tq, tc)
    s = seq_ratio(nq, nc)
    jw = jaro_winkler(nq, nc)
    rf = rapidfuzz_ratio(nq, nc)
    base = 0.50 * j + 0.15 * s + 0.15 * jw + 0.20 * rf
    if nq in nc or nc in nq:
        base = max(base, 0.85)
    return max(0.0, min(1.0, base))


# ---------------- CSV + indices ----------------

def load_items_from_csv(path: str = CSV_PATH) -> List[Dict]:
    """Load items from CSV, supporting aliases/alt_names columns."""
    items: List[Dict] = []
    if not os.path.exists(path):
        logger.warning("CSV not found: %s", path)
        return items
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            aliases_raw = row.get("aliases") or row.get("alt_names") or ""
            aliases = [a.strip() for a in aliases_raw.split(";") if a.strip()]
            items.append({
                "id": (row.get("id") or "").strip(),
                "name": (row.get("name") or "").strip(),
                "country": (row.get("country") or "").strip(),
                "region": (row.get("region") or "").strip(),
                "type": (row.get("type") or "").strip(),
                "aliases": aliases,
                "description": (row.get("description") or "").strip()
            })
    logger.info("Loaded %d items from %s", len(items), path)
    return items


def save_items_to_csv(path: str = CSV_PATH) -> None:
    """Persist current ITEMS back to CSV."""
    fieldnames = ["id", "name", "country", "region", "type", "aliases", "description"]
    try:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for it in ITEMS:
                row = {
                    "id": it.get("id", ""),
                    "name": it.get("name", ""),
                    "country": it.get("country", ""),
                    "region": it.get("region", ""),
                    "type": it.get("type", ""),
                    "aliases": ";".join(it.get("aliases", [])),
                    "description": it.get("description", ""),
                }
                writer.writerow(row)
        logger.info("Saved %d items to %s", len(ITEMS), path)
    except Exception as e:
        logger.exception("Failed to save items to CSV %s: %s", path, e)


def build_prefix_index(items: List[Dict], max_prefix_len: int = 6) -> None:
    """Prefix index over normalized name + aliases for fast narrowing."""
    global _PREFIX_INDEX
    _PREFIX_INDEX = {}
    for it in items:
        for cand in [it.get("name", "")] + it.get("aliases", []):
            n = normalize_text(cand)
            if not n:
                continue
            for L in range(1, min(len(n), max_prefix_len) + 1):
                pref = n[:L]
                _PREFIX_INDEX.setdefault(pref, []).append(it)
    logger.debug("Built prefix index with %d keys", len(_PREFIX_INDEX))


def build_blocking_indices(items: List[Dict]) -> None:
    """
    Build country/region/type blocking and token-prefix blocking.
    These are used to narrow candidate sets efficiently.
    """
    global _COUNTRY_INDEX, _REGION_INDEX, _TYPE_INDEX, _TOKEN_PREFIX_INDEX
    _COUNTRY_INDEX = {}
    _REGION_INDEX = {}
    _TYPE_INDEX = {}
    _TOKEN_PREFIX_INDEX = {}
    for it in items:
        cid = it.get("id")
        if not cid:
            continue
        c = normalize_text(it.get("country", ""))
        r = normalize_text(it.get("region", ""))
        t = normalize_text(it.get("type", ""))
        if c:
            _COUNTRY_INDEX.setdefault(c, []).append(cid)
        if r:
            _REGION_INDEX.setdefault(r, []).append(cid)
        if t:
            _TYPE_INDEX.setdefault(t, []).append(cid)
        for cand in [it.get("name", "")] + it.get("aliases", []):
            for tok in tokens(cand):
                key = tok[:3] if len(tok) >= 3 else tok
                if key:
                    _TOKEN_PREFIX_INDEX.setdefault(key, []).append(cid)
    logger.debug(
        "Built blocking indices: countries=%d regions=%d types=%d token_prefixes=%d",
        len(_COUNTRY_INDEX),
        len(_REGION_INDEX),
        len(_TYPE_INDEX),
        len(_TOKEN_PREFIX_INDEX),
    )

# ---------------- Embeddings + Annoy ----------------

def get_local_embedder():
    """Lazy-load local sentence-transformers model."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        if 'SentenceTransformer' not in globals() or SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it or set USE_LOCAL_EMBEDDINGS=false to use pseudo-embeddings."
            )
        _LOCAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _LOCAL_MODEL


def compute_embeddings_for_texts(texts: List[str]) -> np.ndarray:
    """Compute embeddings in batches for all item texts."""
    model = get_local_embedder()
    all_embs = []
    for i in range(0, len(texts), BUILD_BATCH_SIZE):
        batch = texts[i:i + BUILD_BATCH_SIZE]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embs.append(np.array(embs, dtype=np.float32))
    arr = np.vstack(all_embs) if all_embs else np.zeros((0, EMBED_DIM), dtype=np.float32)
    return arr


def safe_save_embeddings(ids: List[str], embeddings: np.ndarray, path: str = EMBEDDINGS_PATH) -> None:
    """Save embeddings only if they look non-empty and non-zero."""
    if embeddings is None or embeddings.size == 0:
        return
    if not np.any(np.abs(embeddings) > 1e-6):
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, ids=np.array(ids), embeddings=embeddings)
        logger.info("Saved embeddings to %s", path)
    except Exception as e:
        logger.exception("Failed to save embeddings to %s: %s", path, e)


def build_item_text(it: Dict) -> str:
    """Concatenate fields into a single text for embedding."""
    parts = [it.get("name", "")]
    parts += it.get("aliases", [])
    parts += [
        it.get("country", ""),
        it.get("region", ""),
        it.get("type", ""),
        it.get("description", "")
    ]
    return " | ".join([p for p in parts if p])


def load_embeddings(path: str = EMBEDDINGS_PATH) -> Tuple[List[str], Optional[np.ndarray]]:
    """Load embeddings from disk if present."""
    if not os.path.exists(path):
        return [], None
    data = np.load(path, allow_pickle=True)
    ids = data["ids"].tolist()
    embeddings = data["embeddings"]
    logger.info("Loaded embeddings from %s (n=%d, dim=%d)", path, len(ids), embeddings.shape[1])
    return ids, embeddings


def build_annoy_index(embeddings: np.ndarray, dim: int, path: str = ANN_INDEX_PATH, trees: int = ANNOY_TREES) -> "AnnoyIndex":
    """Build and persist Annoy index for fast approximate NN search."""
    if not _HAS_ANNOY:
        raise RuntimeError("annoy not installed")
    idx = AnnoyIndex(dim, metric=ANNOY_METRIC)
    for i in range(embeddings.shape[0]):
        idx.add_item(i, embeddings[i].astype(np.float32))
    idx.build(trees)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        idx.save(path)
    except Exception as e:
        logger.exception("Failed to save Annoy index to %s: %s", path, e)
    logger.info("Built and saved Annoy index to %s", path)
    return idx


def load_annoy_index(dim: int, path: str = ANN_INDEX_PATH) -> Optional["AnnoyIndex"]:
    """Load Annoy index from disk if present."""
    if not _HAS_ANNOY:
        return None
    if not os.path.exists(path):
        return None
    idx = AnnoyIndex(dim, metric=ANNOY_METRIC)
    idx.load(path)
    logger.info("Loaded Annoy index from %s", path)
    return idx


def rebuild_after_mutation() -> None:
    """
    After admin add/update/delete:
    - rebuild indices
    - recompute embeddings
    - rebuild Annoy
    - persist CSV + embeddings + index
    """
    global _EMBEDDINGS, _ANN_INDEX, _ID_TO_INDEX, EMBED_DIM

    build_prefix_index(ITEMS)
    build_blocking_indices(ITEMS)
    _ID_TO_INDEX = {it.get("id"): i for i, it in enumerate(ITEMS)}

    save_items_to_csv(CSV_PATH)

    if SKIP_EMBEDDINGS_ON_START:
        _EMBEDDINGS = None
        _ANN_INDEX = None
        return

    texts = [build_item_text(it) for it in ITEMS]
    ids = [it.get("id") for it in ITEMS]
    if texts and _USE_LOCAL:
        try:
            emb = compute_embeddings_for_texts(texts)
        except Exception as e:
            logger.exception("Failed to recompute embeddings after mutation: %s", e)
            _EMBEDDINGS = None
            _ANN_INDEX = None
        else:
            _EMBEDDINGS = emb
            EMBED_DIM = emb.shape[1]
            safe_save_embeddings(ids, emb)
            if _HAS_ANNOY and _EMBEDDINGS.shape[0] > 0:
                try:
                    _ANN_INDEX = build_annoy_index(_EMBEDDINGS, _EMBEDDINGS.shape[1])
                except Exception as e:
                    logger.exception("Failed to rebuild Annoy index after mutation: %s", e)
                    _ANN_INDEX = None
    else:
        _EMBEDDINGS = None
        _ANN_INDEX = None


def initialize_all() -> None:
    """Load items, build indices, and prepare embeddings + Annoy."""
    global ITEMS, _EMBEDDINGS, _ANN_INDEX, _ID_TO_INDEX, EMBED_DIM

    ITEMS = load_items_from_csv(CSV_PATH)
    build_prefix_index(ITEMS)
    build_blocking_indices(ITEMS)

    if SKIP_EMBEDDINGS_ON_START:
        _EMBEDDINGS = None
    else:
        ids, embeddings = load_embeddings(EMBEDDINGS_PATH)
        if embeddings is None or len(ids) != len(ITEMS):
            texts = [build_item_text(it) for it in ITEMS]
            ids = [it.get("id") for it in ITEMS]
            if texts and _USE_LOCAL:
                try:
                    emb = compute_embeddings_for_texts(texts)
                except Exception as e:
                    logger.exception("Failed to compute embeddings: %s", e)
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

    if _EMBEDDINGS is not None and _EMBEDDINGS.shape[0] > 0 and _HAS_ANNOY:
        try:
            idx = load_annoy_index(_EMBEDDINGS.shape[1])
            if idx is None or idx.get_n_items() != _EMBEDDINGS.shape[0]:
                idx = build_annoy_index(_EMBEDDINGS, _EMBEDDINGS.shape[1])
            _ANN_INDEX = idx
        except Exception as e:
            logger.exception("Failed to build/load Annoy index: %s", e)
            _ANN_INDEX = None
    else:
        _ANN_INDEX = None

    _ID_TO_INDEX = {it.get("id"): i for i, it in enumerate(ITEMS)}

    logger.info(
        "Initialization complete: items=%d embeddings=%s annoy=%s",
        len(ITEMS),
        "yes" if _EMBEDDINGS is not None else "no",
        "yes" if _ANN_INDEX is not None else "no",
    )

# ---------------- Query embeddings ----------------

@cached(QUERY_EMBED_CACHE)
def get_query_embedding(text: str) -> np.ndarray:
    """Get (cached) embedding for a query string."""
    if _USE_LOCAL:
        model = get_local_embedder()
        emb = model.encode([text], convert_to_numpy=True)[0]
        return np.array(emb, dtype=np.float32)
    return np.array([float(sum(ord(c) for c in text) % 1000)] * EMBED_DIM, dtype=np.float32)


# ---------------- Scoring + calibration ----------------

def best_name_match_score(query: str, item: Dict) -> Tuple[float, str]:
    """Best name/alias score for an item, with small alias bonus."""
    best = combined_name_score(query, item.get("name", ""))
    best_str = item.get("name", "")
    for alt in item.get("aliases", []):
        s = combined_name_score(query, alt)
        if s > best:
            best = s
            best_str = alt
    if best_str != item.get("name", "") and best > 0:
        best = min(1.0, best + 0.05)
    return best, best_str


def field_match_boost(query_tokens: List[str], field_value: str) -> float:
    """Simple overlap-based boost for country/region/type fields."""
    if not field_value:
        return 0.0
    ftoks = tokens(field_value)
    if not ftoks:
        return 0.0
    inter = set(query_tokens).intersection(set(ftoks))
    if not inter:
        return 0.0
    return len(inter) / len(ftoks)


def string_item_score(query: str, item: Dict) -> Tuple[float, Dict]:
    """Compute string-based score for an item."""
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

    capital_bonus = 0.05 if is_capital(item) else 0.0

    score = (
        0.65 * name_score
        + 0.15 * country_boost
        + 0.12 * region_boost
        + 0.08 * type_boost
        + capital_bonus
    )
    score = max(0.0, min(1.0, score))

    details = {
        "name_score": round(name_score, 4),
        "country_boost": round(country_boost, 4),
        "region_boost": round(region_boost, 4),
        "type_boost": round(type_boost, 4),
        "matched_string": matched_str,
    }
    return score, details


def embed_candidates(query_emb: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
    """Return (index, cosine_sim) candidates from Annoy, if available."""
    if _ANN_INDEX is None or _EMBEDDINGS is None or _EMBEDDINGS.shape[0] == 0:
        return []
    with _ANN_LOCK:
        try:
            idxs, dists = _ANN_INDEX.get_nns_by_vector(
                query_emb.astype(np.float32), top_k, include_distances=True
            )
            sims = []
            for dist in dists:
                if ANNOY_METRIC == "angular":
                    cos_sim = max(-1.0, min(1.0, 1.0 - (dist ** 2) / 2.0))
                else:
                    cos_sim = 1.0 - dist
                sims.append(float(cos_sim))
            return list(zip(idxs, sims))
        except Exception:
            return []


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Standard cosine similarity."""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def hybrid_score_for_item(query: str, query_emb: np.ndarray, item: Dict, item_index: Optional[int]):
    """Combine string score and embedding similarity."""
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
        **s_details,
    }
    return final, details


def calibrate_score(raw_score: float, name_score: float) -> float:
    """Confidence calibration."""
    s = raw_score
    if name_score >= 0.95:
        s = min(1.0, s + 0.1)
    elif name_score < 0.4:
        s = max(0.0, s - 0.1)
    if s < 0.4:
        s *= 0.8
    elif s > 0.8:
        s = 0.8 + (s - 0.8) * 1.2
    return max(0.0, min(1.0, s))


# ---------------- LLM-style fuzzy matcher ----------------

def llm_fuzzy_matches(query: str, limit: int = 5):
    """Semantic fuzzy matcher."""
    if not query:
        return []
    if _EMBEDDINGS is None or _EMBEDDINGS.shape[0] == 0:
        return find_matches(query, limit=limit, cutoff=0.0)

    nq = normalize_text(query)
    q_toks = tokens(nq)
    query_emb = get_query_embedding(query)

    neighbors = embed_candidates(query_emb, top_k=max(limit * 5, 20))
    neighbor_indices = {idx for idx, _ in neighbors}

    strong_lexical_indices = set()
    for i, it in enumerate(ITEMS):
        name_score, _ = best_name_match_score(query, it)
        if name_score >= 0.8:
            strong_lexical_indices.add(i)

    candidate_indices = neighbor_indices.union(strong_lexical_indices)

    results = []
    for idx in candidate_indices:
        if idx < 0 or idx >= len(ITEMS):
            continue
        item = ITEMS[idx]
        string_score, s_details = string_item_score(query, item)

        all_name_tokens = tokens(item.get("name", ""))
        for alt in item.get("aliases", []):
            all_name_tokens += tokens(alt)
        j = jaccard(q_toks, all_name_tokens)

        if s_details.get("name_score", 0.0) < 0.2 and j < 0.05:
            continue

        if idx in neighbor_indices:
            cos_sim = next((cs for i2, cs in neighbors if i2 == idx), 0.0)
        else:
            item_emb = _EMBEDDINGS[idx]
            cos_sim = cosine_similarity(query_emb, item_emb)

        embed_score = max(0.0, (cos_sim + 1.0) / 2.0)

        raw = LLM_WEIGHT * embed_score + (1.0 - LLM_WEIGHT) * string_score
        final = calibrate_score(raw, s_details.get("name_score", 0.0))

        details = {
            "llm_embed_score": round(embed_score, 4),
            "string_score": round(string_score, 4),
            "calibrated_score": round(final, 4),
            **s_details,
        }
        results.append((final, item, details))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:limit]


# ---------------- ANN-only matcher ----------------

def ann_only_matches(query: str, limit: int = 5):
    """Return candidates ranked purely by embedding similarity (ANN neighbors)."""
    if not query:
        return []
    if _EMBEDDINGS is None or _EMBEDDINGS.shape[0] == 0 or _ANN_INDEX is None:
        return []
    q_emb = get_query_embedding(query)
    neighbors = embed_candidates(q_emb, top_k=max(limit * 5, 20))
    results = []
    for idx, sim in neighbors:
        if idx < 0 or idx >= len(ITEMS):
            continue
        item = ITEMS[idx]
        # convert sim (cosine) to 0..1
        embed_score = max(0.0, (sim + 1.0) / 2.0) if sim <= 1.0 else sim
        # small string score to break ties
        string_score, s_details = string_item_score(query, item)
        raw = EMBED_WEIGHT * embed_score + (1.0 - EMBED_WEIGHT) * string_score
        final = calibrate_score(raw, s_details.get("name_score", 0.0))
        details = {"embed_score": round(embed_score, 4), "string_score": round(string_score, 4), **s_details}
        results.append((final, item, details))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:limit]


# ---------------- Hybrid matcher ----------------

def find_matches(query: str, limit: int = 5, cutoff: float = MATCH_THRESHOLD):
    """Main hybrid matcher."""
    if not query:
        return []

    nq = normalize_text(query)
    q_toks = tokens(nq)
    candidates = set()

    # Prefix narrowing
    for L in range(len(nq), 0, -1):
        key = nq[:L]
        if key in _PREFIX_INDEX:
            for it in _PREFIX_INDEX[key]:
                cid = it.get("id")
                if cid:
                    candidates.add(cid)
            break

    # Country/region/type blocking
    for tok in q_toks:
        nt = normalize_text(tok)
        if nt in _COUNTRY_INDEX:
            candidates.update(_COUNTRY_INDEX[nt])
        if nt in _REGION_INDEX:
            candidates.update(_REGION_INDEX[nt])
        if nt in _TYPE_INDEX:
            candidates.update(_TYPE_INDEX[nt])

    # Token-prefix blocking
    for tok in q_toks:
        key = tok[:3] if len(tok) >= 3 else tok
        if key in _TOKEN_PREFIX_INDEX:
            candidates.update(_TOKEN_PREFIX_INDEX[key])

    if not candidates:
        candidates = {it.get("id") for it in ITEMS if it.get("id")}

    query_emb = get_query_embedding(query)
    embed_neigh = embed_candidates(query_emb, top_k=limit * 3)
    embed_candidate_ids = {
        ITEMS[idx].get("id") for idx, _ in embed_neigh if 0 <= idx < len(ITEMS)
    }
    candidates = candidates.union(embed_candidate_ids)

    scored = []
    for cid in candidates:
        idx = _ID_TO_INDEX.get(cid)
        if idx is None:
            continue
        item = ITEMS[idx]
        raw_score, details = hybrid_score_for_item(query, query_emb, item, idx)
        calibrated = calibrate_score(raw_score, details.get("name_score", 0.0))
        details["calibrated_score"] = round(calibrated, 4)

        if calibrated >= cutoff:
            scored.append((calibrated, item, details))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]


# ---------------- Suggest ----------------

def suggest_entities(prefix: str, limit: int = 10):
    """Autocomplete suggestions."""
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
        cid = it.get("id")
        if not cid or cid in seen:
            continue
        seen.add(cid)

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
            "type": [{"id": "Place", "name": "Place"}],
        })
    return out


# ---------------- Improvements ----------------

def explain_improvement(query: str, item: Dict, details: Dict) -> str:
    """Human-readable explanation for improvement suggestions."""
    name = item.get("name", "")
    country = item.get("country", "")
    region = item.get("region", "")
    nq = normalize_text(query)

    reasons = []
    if details.get("name_score", 0.0) > 0.8:
        reasons.append(f"very similar name to '{query}'")
    elif details.get("name_score", 0.0) > 0.6:
        reasons.append("similar spelling")

    if country and normalize_text(country) in nq:
        reasons.append(f"matches country '{country}' mentioned in the query")
    elif country:
        reasons.append(f"located in {country}")

    if region and normalize_text(region) in nq:
        reasons.append(f"matches region '{region}' mentioned in the query")

    if not reasons:
        reasons.append("semantically similar based on description and metadata")

    return f"{name} is suggested because it is " + " and ".join(reasons) + "."


def suggest_improvements_for_query(query: str, limit: int = 5):
    """Suggest improvements with explanations."""
    matches = find_matches(query, limit=limit * 2, cutoff=0.0)
    if not matches:
        return []
    suggestions = []
    for score, item, details in matches:
        if score < 0.3:
            continue
        reason = explain_improvement(query, item, details)
        suggestions.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "score": int(round(score * 100)),
            "match": score >= MATCH_THRESHOLD,
            "metadata": {
                "country": item.get("country", ""),
                "region": item.get("region", ""),
                "type": item.get("type", ""),
            },
            "reason": reason,
            "details": details,
        })
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    return suggestions[:limit]


# ---------------- Admin helpers ----------------

def require_admin():
    token = request.headers.get("X-ADMIN-TOKEN") or request.args.get("admin_token")
    if not ADMIN_TOKEN:
        abort(403, description="Admin token not configured on server")
    if not token or token != ADMIN_TOKEN:
        abort(401, description="Unauthorized")


# ---------------- Routes ----------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "items": len(ITEMS),
        "embeddings": _EMBEDDINGS is not None,
        "annoy": _ANN_INDEX is not None,
    })


@app.route("/service", methods=["GET", "POST"])
def service_metadata():
    return jsonify({
        "name": "Places Reconciliation (Hybrid + Calibrated LLM Fuzzy)",
        "identifierSpace": "http://example.org/places",
        "schemaSpace": "http://example.org/schema",
        "view": {"url": f"http://{HOST}:{PORT}/view/{{id}}"},
        "defaultTypes": [{"id": "Place", "name": "Place"}],
        "preview": {"url": f"http://{HOST}:{PORT}/view/{{id}}", "width": 400, "height": 200},
        "properties": [
            {"id": "country", "name": "Country"},
            {"id": "region", "name": "Region"},
            {"id": "type", "name": "Type"},
        ],
    })


@app.route("/view/<entity_id>", methods=["GET"])
def view_entity(entity_id):
    item = next((i for i in ITEMS if i.get("id") == entity_id), None)
    if not item:
        abort(404)
    return jsonify(item)


# ---------------- Reconcile (GET + POST) ----------------

@app.route("/reconcile", methods=["GET", "POST"])
def reconcile():
    # GET single query
    if request.method == "GET":
        query = request.args.get("query", "").strip()
        limit = int(request.args.get("limit", 5))
        cutoff = float(request.args.get("cutoff", MATCH_THRESHOLD))
        matches = find_matches(query, limit=limit, cutoff=cutoff)
        result = []
        for score, item, details in matches:
            match_flag = bool(details.get("name_score", 0.0) >= 0.95 and score >= 0.6)
            result.append({
                "id": item.get("id"),
                "name": item.get("name"),
                "score": int(round(score * 100)),
                "match": match_flag,
                "type": [{"id": "Place", "name": "Place"}],
                "description": item.get("description", ""),
                "metadata": {
                    "country": item.get("country", ""),
                    "region": item.get("region", ""),
                    "type": item.get("type", ""),
                },
                "details": details,
            })
        return jsonify({"result": result})

    # POST batch mode
    payload = request.get_json(silent=True) or {}
    queries = payload.get("queries", {})
    out = {}
    for qid, qobj in queries.items():
        qtext = qobj.get("query", "")
        limit = int(qobj.get("limit", 5)) if isinstance(qobj, dict) and "limit" in qobj else 5
        cutoff = float(qobj.get("cutoff", MATCH_THRESHOLD)) if isinstance(qobj, dict) and "cutoff" in qobj else MATCH_THRESHOLD
        matches = find_matches(qtext, limit=limit, cutoff=cutoff)
        result = []
        for score, item, details in matches:
            match_flag = bool(details.get("name_score", 0.0) >= 0.95 and score >= 0.6)
            result.append({
                "id": item.get("id"),
                "name": item.get("name"),
                "score": int(round(score * 100)),
                "match": match_flag,
                "type": [{"id": "Place", "name": "Place"}],
                "description": item.get("description", ""),
                "metadata": {
                    "country": item.get("country", ""),
                    "region": item.get("region", ""),
                    "type": item.get("type", ""),
                },
                "details": details,
            })
        out[qid] = {"result": result}
    return jsonify(out)


# ---------------- Compare endpoint (A: comparative modes) ----------------

@app.route("/compare", methods=["GET"])
def compare_matchers():
    """
    Compare different matching strategies for a single query.
    Returns results from:
      - lexical-only (string_item_score)
      - hybrid (find_matches)
      - llm_fuzzy (llm_fuzzy_matches)
      - ann-only (ann_only_matches)
    """
    query = request.args.get("query", "").strip()
    limit = int(request.args.get("limit", 5))
    if not query:
        return jsonify({"error": "query parameter required"}), 400

    # Lexical-only: compute string_item_score across candidates (fast blocking)
    lexical_candidates = set()
    nq = normalize_text(query)
    for L in range(len(nq), 0, -1):
        key = nq[:L]
        if key in _PREFIX_INDEX:
            for it in _PREFIX_INDEX[key]:
                cid = it.get("id")
                if cid:
                    lexical_candidates.add(cid)
            break
    if not lexical_candidates:
        lexical_candidates = {it.get("id") for it in ITEMS if it.get("id")}
    lexical_results = []
    for cid in lexical_candidates:
        idx = _ID_TO_INDEX.get(cid)
        if idx is None:
            continue
        item = ITEMS[idx]
        s_score, s_details = string_item_score(query, item)
        calibrated = calibrate_score(s_score, s_details.get("name_score", 0.0))
        lexical_results.append((calibrated, item, s_details))
    lexical_results.sort(key=lambda x: x[0], reverse=True)
    lexical_out = [{
        "id": it.get("id"), "name": it.get("name"), "score": int(round(s*100)), "details": d
    } for s, it, d in lexical_results[:limit]]

    # Hybrid
    hybrid = find_matches(query, limit=limit, cutoff=0.0)
    hybrid_out = [{
        "id": it.get("id"), "name": it.get("name"), "score": int(round(s*100)), "details": d
    } for s, it, d in hybrid]

    # LLM fuzzy
    llm = llm_fuzzy_matches(query, limit=limit)
    llm_out = [{
        "id": it.get("id"), "name": it.get("name"), "score": int(round(s*100)), "details": d
    } for s, it, d in llm]

    # ANN-only
    ann = ann_only_matches(query, limit=limit)
    ann_out = [{
        "id": it.get("id"), "name": it.get("name"), "score": int(round(s*100)), "details": d
    } for s, it, d in ann]

    return jsonify({
        "query": query,
        "lexical": lexical_out,
        "hybrid": hybrid_out,
        "llm": llm_out,
        "ann": ann_out,
    })


# ---------------- Suggest endpoints ----------------

@app.route("/suggest", methods=["GET"])
def suggest_route():
    q = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 10))
    out = suggest_entities(q, limit=limit)
    return jsonify({"result": out})


@app.route("/suggest_improvements", methods=["GET"])
def suggest_improvements_route():
    q = request.args.get("query", "").strip()
    limit = int(request.args.get("limit", 5))
    out = suggest_improvements_for_query(q, limit=limit)
    return jsonify({"result": out})


# ---------------- Data extension endpoint (E) ----------------

@app.route("/extend", methods=["POST"])
def extend_route():
    """
    OpenRefine-style data extension endpoint.
    Accepts JSON:
      {"ids": ["1","2"], "properties": ["country","region","type"]}
    Returns:
      {"rows": {"1": {"country": "...", "region": "..."}, "2": {...}}}
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON"}), 400
    ids = payload.get("ids", [])
    props = payload.get("properties", [])
    if not isinstance(ids, list) or not isinstance(props, list):
        return jsonify({"error": "ids and properties must be lists"}), 400

    rows = {}
    for eid in ids:
        item = next((i for i in ITEMS if i.get("id") == str(eid)), None)
        if not item:
            rows[str(eid)] = {}
            continue
        row = {}
        for p in props:
            # only allow known properties
            if p in ("country", "region", "type", "description", "name", "aliases"):
                val = item.get(p, "")
                # aliases -> return as list
                if p == "aliases":
                    row[p] = item.get("aliases", [])
                else:
                    row[p] = val
        rows[str(eid)] = row
    return jsonify({"rows": rows})


# ---------------- CRUD endpoints ----------------

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
        "aliases": payload.get("aliases") if isinstance(payload.get("aliases"), list)
        else [a.strip() for a in (payload.get("aliases") or "").split(";") if a.strip()],
        "description": str(payload.get("description") or ""),
    }
    if not new["id"] or not new["name"]:
        abort(400, description="id and name required")
    ITEMS.append(new)
    rebuild_after_mutation()
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
        item["aliases"] = payload.get("aliases") if isinstance(payload.get("aliases"), list) \
            else [a.strip() for a in (payload.get("aliases") or "").split(";") if a.strip()]
    rebuild_after_mutation()
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
    rebuild_after_mutation()
    return jsonify({"status": "deleted", "id": entity_id})


@app.route("/_reload_items", methods=["POST"])
def _reload_items():
    global ITEMS
    ITEMS = load_items_from_csv(CSV_PATH)
    build_prefix_index(ITEMS)
    build_blocking_indices(ITEMS)
    t = threading.Thread(target=rebuild_after_mutation, daemon=True)
    t.start()
    return jsonify({"status": "reloading", "count": len(ITEMS)})


# ---------------- Explain endpoint ----------------

@app.route("/explain/<entity_id>", methods=["GET"])
def explain_entity(entity_id):
    """Return explanation of why an entity might match a given query (query param)."""
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "query parameter required"}), 400
    item = next((i for i in ITEMS if i.get("id") == entity_id), None)
    if not item:
        return jsonify({"error": "not found"}), 404
    _, details = string_item_score(query, item)
    explanation = explain_improvement(query, item, details)
    return jsonify({"id": entity_id, "name": item.get("name"), "explanation": explanation, "details": details})


# ---------------- LLM reconcile route (FIX) ----------------

@app.route("/llm_reconcile", methods=["GET"])
def llm_reconcile_route():
    """
    Expose the LLM-style fuzzy matcher as a reconcile-style endpoint.
    Returns JSON in the same shape as /reconcile for a single query.
    """
    query = request.args.get("query", "").strip()
    limit = int(request.args.get("limit", 5))
    if not query:
        return jsonify({"result": []})
    results = llm_fuzzy_matches(query, limit=limit)
    out = []
    for score, item, details in results:
        match_flag = bool(details.get("name_score", 0.0) >= 0.95 and score >= 0.6)
        out.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "score": int(round(score * 100)),
            "match": match_flag,
            "type": [{"id": "Place", "name": "Place"}],
            "description": item.get("description", ""),
            "metadata": {
                "country": item.get("country", ""),
                "region": item.get("region", ""),
                "type": item.get("type", ""),
            },
            "details": details,
        })
    return jsonify({"result": out})


# ---------------- Startup ----------------

if __name__ == "__main__":
    initialize_all()
    app.run(host=HOST, port=PORT, debug=DEBUG)
