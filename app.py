# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import unicodedata
import re
import json
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

app = Flask(__name__)
CORS(app)

CSV_PATH = 'places.csv'

# ---------- Utilities ----------

def normalize_text(s: str) -> str:
    if s is None:
        return ''
    s = s.strip().lower()
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

def combined_score(query: str, candidate: str) -> float:
    """
    Combine token Jaccard and sequence ratio.
    Weigh token overlap higher for multi-word names.
    If the normalized query is a substring or prefix of the candidate,
    return a boosted score to handle abbreviations/short forms.
    Returns value in [0.0, 1.0].
    """
    nq = normalize_text(query)
    nc = normalize_text(candidate)
    if not nq or not nc:
        return 0.0
    # exact equality
    if nq == nc:
        return 1.0

    # If one string is contained in the other (covers abbreviations, short forms),
    # compute base score and ensure a conservative boosted minimum.
    if nq in nc or nc in nq:
        t_q = tokens(nq)
        t_c = tokens(nc)
        j = jaccard(t_q, t_c)
        s = seq_ratio(nq, nc)
        weight_j = 0.6 if (len(t_q) > 1 or len(t_c) > 1) else 0.4
        weight_s = 1.0 - weight_j
        base = weight_j * j + weight_s * s
        # Boost to a reasonable floor for substring/prefix matches
        return max(base, 0.85)

    # Default combined scoring
    t_q = tokens(nq)
    t_c = tokens(nc)
    j = jaccard(t_q, t_c)
    s = seq_ratio(nq, nc)
    weight_j = 0.6 if (len(t_q) > 1 or len(t_c) > 1) else 0.4
    weight_s = 1.0 - weight_j
    return weight_j * j + weight_s * s

# ---------- Load CSV ----------

def load_items_from_csv(path: str = CSV_PATH) -> List[Dict]:
    items = []
    with open(path, encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            alt_raw = row.get('alt_names', '') or ''
            alt_list = [a.strip() for a in alt_raw.split(';') if a.strip()]
            items.append({
                'id': (row.get('id') or '').strip(),
                'name': (row.get('name') or '').strip(),
                'alt_names': alt_list,
                'description': (row.get('description') or '').strip()
            })
    return items

ITEMS = load_items_from_csv()

# ---------- Matching ----------

def best_score_for_item(query: str, item: Dict) -> Tuple[float, str]:
    best = 0.0
    best_str = item.get('name','')
    s = combined_score(query, item.get('name',''))
    if s > best:
        best = s; best_str = item.get('name','')
    for alt in item.get('alt_names', []):
        s2 = combined_score(query, alt)
        if s2 > best:
            best = s2; best_str = alt
    return best, best_str

def find_matches(query: str, limit: int = 5, cutoff: float = 0.55) -> List[Tuple[float, Dict, str]]:
    if not query:
        return []
    scored = []
    for item in ITEMS:
        score, matched_str = best_score_for_item(query, item)
        if score >= cutoff:
            scored.append((score, item, matched_str))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]

# ---------- API endpoints ----------

@app.route('/service', methods=['GET', 'POST'])
def service_metadata():
    return jsonify({
        "name": "Places Reconciliation Prototype (CSV)",
        "identifierSpace": "http://example.org/places",
        "schemaSpace": "http://example.org/schema",
        "view": {"url": "http://127.0.0.1:5000/view/{{id}}"},
        "defaultTypes": [{"id":"Place","name":"Place"}]
    })

@app.route('/view/<entity_id>', methods=['GET'])
def view_entity(entity_id):
    item = next((i for i in ITEMS if i.get('id') == entity_id), None)
    if not item:
        return "Not found", 404
    html = f"<h1>{item.get('name')}</h1><p>{item.get('description','')}</p>"
    if item.get('alt_names'):
        html += "<p><strong>Alternate names:</strong> " + ", ".join(item['alt_names']) + "</p>"
    return html

@app.route('/reconcile', methods=['GET', 'POST'])
def reconcile():
    """
    Robust handling:
    - JSON body with {"queries": {...}}  (preferred)
    - Form-encoded 'queries' field containing JSON string (OpenRefine sometimes sends this)
    - Single query via JSON or GET
    """
    payload = request.get_json(silent=True)

    # If payload is None, check for form-encoded 'queries' parameter
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

    # Batch mode: payload['queries'] expected to be a dict with keys q0, q1, ...
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
                    "match": m[0] == 1.0,
                    "type": [{"id":"Place","name":"Place"}],
                    "description": m[1].get('description','')
                } for m in matches
            ]}
        return jsonify(out)

    # Single query handling (JSON or GET)
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
            "match": m[0] == 1.0,
            "type": [{"id":"Place","name":"Place"}],
            "description": m[1].get('description','')
        } for m in matches
    ]})

# ---------- Run ----------

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)



