# evaluate.py
"""
Evaluate the reconciliation service running at http://127.0.0.1:5000/reconcile
Reads test_inputs.csv (columns: input,expected_id) and computes:
- precision@1
- precision@5
- coverage (fraction with at least one candidate)
Outputs a CSV of failures: evaluation_failures.csv
"""
import csv
import json
import math
import requests
from collections import defaultdict

SERVICE_URL = "http://127.0.0.1:5000/reconcile"
TEST_CSV = "test_inputs.csv"
FAILURES_CSV = "evaluation_failures.csv"
BATCH_SIZE = 20  # number of queries per batch request

def load_tests(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            inp = r.get("input","").strip()
            exp = r.get("expected_id","").strip()
            if inp:
                rows.append((inp, exp if exp else "NONE"))
    return rows

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def call_batch(queries):
    """
    queries: list of input strings
    returns: dict mapping q0.. to result arrays
    """
    payload = {"queries": {}}
    for i, q in enumerate(queries):
        payload["queries"][f"q{i}"] = {"query": q, "limit": 5}
    resp = requests.post(SERVICE_URL, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()

def evaluate():
    tests = load_tests(TEST_CSV)
    total = len(tests)
    top1_correct = 0
    top5_correct = 0
    coverage_count = 0
    failures = []

    for batch in chunked(tests, BATCH_SIZE):
        inputs = [b[0] for b in batch]
        expected = [b[1] for b in batch]
        resp = call_batch(inputs)
        # resp keys are q0, q1, ...
        for i, (inp, exp) in enumerate(batch):
            key = f"q{i}"
            results = resp.get(key, {}).get("result", [])
            ids = [r.get("id") for r in results]
            if ids:
                coverage_count += 1
            if ids and exp != "NONE" and ids[0] == exp:
                top1_correct += 1
            if ids and exp != "NONE" and exp in ids[:5]:
                top5_correct += 1
            # record failures for manual inspection
            if exp == "NONE":
                # expected no match but got candidates
                if ids:
                    failures.append({
                        "input": inp,
                        "expected": exp,
                        "top_ids": ";".join([str(x) for x in ids]),
                        "top_names": ";".join([r.get("name","") for r in results]),
                        "scores": ";".join([str(r.get("score","")) for r in results])
                    })
            else:
                if not ids or exp not in ids[:5]:
                    failures.append({
                        "input": inp,
                        "expected": exp,
                        "top_ids": ";".join([str(x) for x in ids]) if ids else "",
                        "top_names": ";".join([r.get("name","") for r in results]) if ids else "",
                        "scores": ";".join([str(r.get("score","")) for r in results]) if ids else ""
                    })

    precision_at_1 = top1_correct / total if total else 0.0
    precision_at_5 = top5_correct / total if total else 0.0
    coverage = coverage_count / total if total else 0.0

    # Print summary
    print("Evaluation summary")
    print("------------------")
    print(f"Total queries: {total}")
    print(f"Precision@1: {precision_at_1:.3f} ({top1_correct}/{total})")
    print(f"Precision@5: {precision_at_5:.3f} ({top5_correct}/{total})")
    print(f"Coverage (>=1 candidate): {coverage:.3f} ({coverage_count}/{total})")
    print(f"Failures recorded: {len(failures)} -> {FAILURES_CSV}")

    # Write failures CSV
    with open(FAILURES_CSV, "w", encoding="utf-8", newline="") as fh:
        fieldnames = ["input","expected","top_ids","top_names","scores"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for f in failures:
            writer.writerow(f)

    # Save a small JSON summary
    summary = {
        "total": total,
        "precision_at_1": precision_at_1,
        "precision_at_5": precision_at_5,
        "coverage": coverage,
        "failures": len(failures)
    }
    with open("evaluation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

if __name__ == "__main__":
    evaluate()
