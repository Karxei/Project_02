# app.py (improved)
"""
Enhanced reconciliation service with:
- multi-field scoring (name, country, region, type)
- improved fuzzy matching (jaro-winkler via jellyfish, RapidFuzz fallback/optional)
- suggest endpoint with simple prefix index
- configurable match threshold and weights
- dev reload route to reload CSV without restarting
- clearer logging for debugging and evaluation
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import unicodedata
import re
import json
import logging
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import os

# Optional libraries
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

# Configuration
CSV_PATH = os.getenv('PLACES_CSV', 'places.csv')
HOST = os.getenv('APP_HOST', '127.0.0.1')
PORT = int(os.getenv('APP_PORT', 5000))
DEBUG = os.getenv('APP_DEBUG', 'true').lower() in ('1', 'true', 'yes')

# Matching configuration
MATCH_THRESHOLD = float(os.getenv('MATCH_THRESHOLD', 0.95))  # when to mark "match": true
DEFAULT_WEIGHTS = {
    'name': float(os.getenv('WEIGHT_NAME', 0.65)),
    'country': float(os.getenv('WEIGHT_COUNTRY', 0.15)),
    'region': float(os.getenv('WEIGHT_REGION', 0.12)),
    'type': float(os.getenv('WEIGHT_TYPE', 0.08))
}
SUGGEST_PREFIX_SCORE = 0.95

# Setup logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# In-memory data
ITEMS: List[Dict] = []
_PREFIX_INDEX: Dict[str, List[Dict]] = {}  # simple prefix -> list of items (for suggest)

# ---------- Utilities ----------

def normalize_text(s: str) -> str:
    if s is None:
        return ''
    s = str(s).strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r'[^\w\s\-\.\,]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokens(s: str) -> List[str]:
    return [t for t in normalize_text(s).split(' ') if t]

def jaccard(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 0.0
    set_a = set(a); set_b = set(b)
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
    else:
        return seq_ratio(a, b)

def rapidfuzz_ratio(a: str, b: str) -> float:
    """
    Use RapidFuzz token_sort_ratio if available for robust fuzzy matching.
    Returns value in [0.0, 1.0].
    """
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        try:
            # token_sort_ratio returns 0-100
            return rf_fuzz.token_sort_ratio(a, b) / 100.0
        except Exception:
            return seq_ratio(a, b)
    else:
        return seq_ratio(a, b)

# Combined name similarity using multiple measures
def combined_name_score(query: str, candidate: str) -> float:
    nq = normalize_text(query)
    nc = normalize_text(candidate)
    if not nq or not nc:
        return 0.0
    if nq == nc:
        return 1.0

    t_q = tokens(nq)
    t_c = tokens(nc)
    j = jaccard(t_q, t_c)
    s = seq_ratio(nq, nc)
    jw = jaro_winkler(nq, nc)
    rf = rapidfuzz_ratio(nq, nc)

    # Combine measures: give token overlap more weight for multi-word names,
    # and include RapidFuzz/jaro as additional signals.
    weight_j = 0.50 if (len(t_q) > 1 or len(t_c) > 1) else 0.30
    weight_s = 0.15
    weight_jw = 0.15
    weight_rf = 1.0 - (weight_j + weight_s + weight_jw)
    base = weight_j * j + weight_s * s + weight_jw * jw + weight_rf * rf

    # substring/prefix boost for abbreviations or short forms
    if nq in nc or nc in nq:
        return max(base, 0.85)

    return max(0.0, min(1.0, base))

# ---------- CSV loading and prefix index ----------

def load_items_from_csv(path: str = CSV_PATH) -> List[Dict]:
    items = []
    if not os.path.exists(path):
        logger.warning("CSV file not found: %s", path)
        return items
    with open(path, encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            alt_raw = row.get('aliases') or row.get('alt_names') or ''
            alt_list = [a.strip() for a in alt_raw.split(';') if a.strip()]
            items.append({
                'id': (row.get('id') or '').strip(),
                'name': (row.get('name') or '').strip(),
                'country': (row.get('country') or '').strip(),
                'region': (row.get('region') or '').strip(),
                'type': (row.get('type') or '').strip(),
                'aliases': alt_list,
                'description': (row.get('description') or '').strip()
            })
    logger.info("Loaded %d items from %s", len(items), path)
    return items

def build_prefix_index(items: List[Dict], max_prefix_len: int = 6):
    """
    Build a simple prefix index mapping normalized prefixes to items.
    Only indexes the primary name and aliases for quick suggest.
    """
    global _PREFIX_INDEX
    _PREFIX_INDEX = {}
    for it in items:
        candidates = [it.get('name','')] + it.get('aliases', [])
        for cand in candidates:
            n = normalize_text(cand)
            # index prefixes up to max_prefix_len characters (or tokens)
            for L in range(1, min(len(n), max_prefix_len) + 1):
                pref = n[:L]
                _PREFIX_INDEX.setdefault(pref, []).append(it)
    logger.debug("Built prefix index with %d keys", len(_PREFIX_INDEX))

# Initialize ITEMS and prefix index
def _init_data():
    global ITEMS
    ITEMS = load_items_from_csv(CSV_PATH)
    build_prefix_index(ITEMS)

_init_data()

# ---------- Matching (multi-field) ----------

def best_name_match_score(query: str, item: Dict) -> Tuple[float, str]:
    best = 0.0
    best_str = item.get('name','')
    s = combined_name_score(query, item.get('name',''))
    if s > best:
        best = s; best_str = item.get('name','')
    for alt in item.get('aliases', []):
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

def item_score(query: str, item: Dict, weights: Dict = None) -> Tuple[float, Dict]:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    nq = normalize_text(query)
    q_toks = tokens(nq)

    name_score, matched_str = best_name_match_score(query, item)

    country_boost = field_match_boost(q_toks, item.get('country',''))
    region_boost = field_match_boost(q_toks, item.get('region',''))
    type_boost = field_match_boost(q_toks, item.get('type',''))

    # stronger boost if exact substring present
    if item.get('country') and normalize_text(item.get('country')) in nq:
        country_boost = max(country_boost, 1.0)
    if item.get('region') and normalize_text(item.get('region')) in nq:
        region_boost = max(region_boost, 1.0)
    if item.get('type') and normalize_text(item.get('type')) in nq:
        type_boost = max(type_boost, 1.0)

    score = (
        weights['name'] * name_score +
        weights['country'] * country_boost +
        weights['region'] * region_boost +
        weights['type'] * type_boost
    )

    score = max(0.0, min(1.0, score))

    details = {
        'name_score': round(name_score, 4),
        'country_boost': round(country_boost, 4),
        'region_boost': round(region_boost, 4),
        'type_boost': round(type_boost, 4)
    }
    return score, {**details, 'matched_string': matched_str}

def find_matches(query: str, limit: int = 5, cutoff: float = 0.55) -> List[Tuple[float, Dict, Dict]]:
    if not query:
        return []
    scored = []
    for item in ITEMS:
        score, details = item_score(query, item)
        if score >= cutoff:
            scored.append((score, item, details))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]

# ---------- Suggest endpoint (prefix search) ----------

def suggest_entities(prefix: str, limit: int = 10) -> List[Dict]:
    if not prefix:
        return []
    npref = normalize_text(prefix)
    ptoks = tokens(npref)
    suggestions = []
    # Use prefix index for quick candidates
    candidates = []
    # try exact prefix keys (progressively shorter)
    for L in range(len(npref), 0, -1):
        key = npref[:L]
        if key in _PREFIX_INDEX:
            candidates.extend(_PREFIX_INDEX[key])
            break
    # fallback to scanning all items if index miss
    if not candidates:
        candidates = ITEMS

    seen_ids = set()
    for item in candidates:
        if item.get('id') in seen_ids:
            continue
        seen_ids.add(item.get('id'))
        name_norm = normalize_text(item.get('name',''))
        score = 0.0
        if name_norm.startswith(npref):
            score = SUGGEST_PREFIX_SCORE
        else:
            name_toks = tokens(name_norm)
            j = jaccard(ptoks, name_toks)
            if j > 0:
                score = 0.6 + 0.4 * j
            else:
                for alt in item.get('aliases', []):
                    alt_norm = normalize_text(alt)
                    if alt_norm.startswith(npref):
                        score = max(score, 0.7)
                    else:
                        alt_toks = tokens(alt_norm)
                        j2 = jaccard(ptoks, alt_toks)
                        if j2 > 0:
                            score = max(score, 0.5 + 0.4 * j2)
        if score > 0:
            suggestions.append((score, item))
    suggestions.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, it in suggestions[:limit]:
        out.append({
            'id': it.get('id'),
            'name': it.get('name'),
            'score': int(round(s * 100)),
            'type': [{'id': 'Place', 'name': 'Place'}]
        })
    return out

# ---------- API endpoints ----------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "items": len(ITEMS)})

@app.route('/service', methods=['GET', 'POST'])
def service_metadata():
    return jsonify({
        "name": "Places Reconciliation (Enhanced)",
        "identifierSpace": "http://example.org/places",
        "schemaSpace": "http://example.org/schema",
        "view": {"url": f"http://{HOST}:{PORT}/view/{{{{id}}}}"},
        "defaultTypes": [{"id":"Place","name":"Place"}],
        "preview": {
            "url": f"http://{HOST}:{PORT}/view/{{{{id}}}}",
            "width": 400,
            "height": 200
        },
        "properties": [
            {"id":"country","name":"Country"},
            {"id":"region","name":"Region"},
            {"id":"type","name":"Type"}
        ]
    })

@app.route('/view/<entity_id>', methods=['GET'])
def view_entity(entity_id):
    item = next((i for i in ITEMS if i.get('id') == entity_id), None)
    if not item:
        return "Not found", 404
    html = f"<h1>{item.get('name')}</h1>"
    if item.get('type'):
        html += f"<p><strong>Type:</strong> {item.get('type')}</p>"
    if item.get('country'):
        html += f"<p><strong>Country:</strong> {item.get('country')}</p>"
    if item.get('region'):
        html += f"<p><strong>Region:</strong> {item.get('region')}</p>"
    if item.get('description'):
        html += f"<p>{item.get('description')}</p>"
    if item.get('aliases'):
        html += "<p><strong>Aliases:</strong> " + ", ".join(item['aliases']) + "</p>"
    return html

@app.route('/reconcile', methods=['GET', 'POST'])
def reconcile():
    payload = request.get_json(silent=True)

    if payload is None:
        if 'queries' in request.form:
            try:
                payload = {'queries': json.loads(request.form['queries'])}
            except Exception:
                payload = None
        else:
            raw = request.get_data(as_text=True)
            if raw:
                try:
                    payload = json.loads(raw)
                except Exception:
                    payload = None

    # Batch mode
    if isinstance(payload, dict) and 'queries' in payload:
        out = {}
        queries_obj = payload['queries']
        if isinstance(queries_obj, str):
            try:
                queries_obj = json.loads(queries_obj)
            except Exception:
                queries_obj = {}
        for key, qobj in queries_obj.items():
            qtext = qobj.get('query','') if isinstance(qobj, dict) else ''
            limit = int(qobj.get('limit', 5)) if isinstance(qobj, dict) and 'limit' in qobj else 5
            cutoff = float(qobj.get('cutoff', 0.55)) if isinstance(qobj, dict) and 'cutoff' in qobj else 0.55
            matches = find_matches(qtext, limit=limit, cutoff=cutoff)
            out[key] = {"result": [
                {
                    "id": m[1].get('id'),
                    "name": m[1].get('name'),
                    "score": int(round(m[0] * 100)),
                    "match": m[0] >= MATCH_THRESHOLD,
                    "type": [{"id":"Place","name":"Place"}],
                    "description": m[1].get('description',''),
                    "metadata": {
                        "country": m[1].get('country',''),
                        "region": m[1].get('region',''),
                        "type": m[1].get('type','')
                    }
                } for m in matches
            ]}
        return jsonify(out)

    # Single query handling
    if isinstance(payload, dict) and 'query' in payload:
        qtext = payload.get('query','')
        limit = int(payload.get('limit', 5))
        cutoff = float(payload.get('cutoff', 0.55))
    else:
        qtext = request.args.get('query','')
        limit = int(request.args.get('limit', 5))
        cutoff = float(request.args.get('cutoff', 0.55))

    matches = find_matches(qtext, limit=limit, cutoff=cutoff)
    return jsonify({"result": [
        {
            "id": m[1].get('id'),
            "name": m[1].get('name'),
            "score": int(round(m[0] * 100)),
            "match": m[0] >= MATCH_THRESHOLD,
            "type": [{"id":"Place","name":"Place"}],
            "description": m[1].get('description',''),
            "metadata": {
                "country": m[1].get('country',''),
                "region": m[1].get('region',''),
                "type": m[1].get('type','')
            },
            "details": m[2]
        } for m in matches
    ]})

@app.route('/suggest', methods=['GET'])
def suggest():
    q = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))
    suggestions = suggest_entities(q, limit=limit)
    return jsonify({"result": suggestions})

# Dev-only: reload items without restarting server
@app.route('/_reload_items', methods=['POST'])
def _reload_items():
    global ITEMS
    ITEMS = load_items_from_csv(CSV_PATH)
    build_prefix_index(ITEMS)
    return jsonify({"status": "reloaded", "count": len(ITEMS)})

# ---------- Run ----------

if __name__ == '__main__':
    try:
        _init_data()
    except Exception as e:
        logger.exception("Failed to initialize data: %s", e)
    app.run(host=HOST, port=PORT, debug=DEBUG)

