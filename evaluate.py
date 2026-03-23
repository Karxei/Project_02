import csv
import json
import time
import requests
from collections import defaultdict, Counter
from statistics import mean

SERVICE = "http://127.0.0.1:5000"

# -----------------------------
# Utility functions
# -----------------------------

def load_test_data(path):
    tests = []
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            q = row.get("query", "").strip()
            exp = row.get("expected_id", "").strip()
            if q:
                tests.append((q, exp))
    return tests


def call_api(endpoint, params=None):
    """Call GET endpoint and return JSON."""
    url = f"{SERVICE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def call_reconcile(query, matcher="hybrid"):
    """
    matcher options:
    - lexical
    - hybrid
    - ann
    - llm
    """
    if matcher == "lexical":
        data = call_api("compare", {"query": query})
        return data.get("lexical", [])
    elif matcher == "ann":
        data = call_api("compare", {"query": query})
        return data.get("ann", [])
    elif matcher == "llm":
        data = call_api("llm_reconcile", {"query": query}).get("result", [])
        return data
    else:
        # default hybrid
        data = call_api("compare", {"query": query})
        return data.get("hybrid", [])


# -----------------------------
# Metrics
# -----------------------------

def precision_at_k(results, expected_id, k):
    if not results:
        return 0
    top_k = [r.get("id") for r in results[:k]]
    return 1 if expected_id in top_k else 0


def recall_at_k(results, expected_id, k):
    if expected_id == "NONE":
        return 1 if not results else 0
    top_k = [r.get("id") for r in results[:k]]
    return 1 if expected_id in top_k else 0


def f1(p, r):
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)


# -----------------------------
# Error classification
# -----------------------------

def classify_error(query, expected_id, results):
    if expected_id == "NONE":
        if results:
            return "False positive"
        return "Correct NONE"

    if not results:
        return "No candidates"

    top1 = results[0].get("id")

    if top1 == expected_id:
        return "Correct"

    # Wrong but similar name?
    top_name = results[0].get("name", "").lower()
    if query.lower() in top_name or top_name in query.lower():
        return "Similar name confusion"

    # Wrong country?
    meta = results[0].get("metadata", {})
    if meta:
        return "Wrong country/region"

    return "Other mismatch"


# -----------------------------
# Threshold sweep
# -----------------------------

def threshold_sweep(tests, matcher, thresholds):
    sweep = []
    for th in thresholds:
        p1_list, r1_list = [], []
        for q, exp in tests:
            results = call_reconcile(q, matcher)
            # filter by threshold
            filtered = [r for r in results if r.get("score", 0) >= int(th * 100)]
            p1_list.append(precision_at_k(filtered, exp, 1))
            r1_list.append(recall_at_k(filtered, exp, 1))
        P = mean(p1_list)
        R = mean(r1_list)
        F = f1(P, R)
        sweep.append((th, P, R, F))
    return sweep


# -----------------------------
# Runtime measurement
# -----------------------------

def measure_runtime(tests, matcher):
    times = []
    for q, _ in tests:
        start = time.time()
        _ = call_reconcile(q, matcher)
        end = time.time()
        times.append(end - start)
    return {
        "avg_ms": mean(times) * 1000,
        "max_ms": max(times) * 1000,
        "min_ms": min(times) * 1000
    }


# -----------------------------
# Per-country breakdown
# -----------------------------

def per_country_metrics(tests, matcher):
    country_stats = defaultdict(lambda: {"p1": [], "r1": []})

    for q, exp in tests:
        results = call_reconcile(q, matcher)
        if results:
            country = results[0].get("metadata", {}).get("country", "Unknown")
        else:
            country = "Unknown"

        country_stats[country]["p1"].append(precision_at_k(results, exp, 1))
        country_stats[country]["r1"].append(recall_at_k(results, exp, 1))

    summary = {}
    for country, vals in country_stats.items():
        P = mean(vals["p1"])
        R = mean(vals["r1"])
        summary[country] = {
            "precision@1": P,
            "recall@1": R,
            "f1@1": f1(P, R)
        }
    return summary


# -----------------------------
# Main evaluation
# -----------------------------

def evaluate_all(test_csv="evaluation_tests.csv"):
    tests = load_test_data(test_csv)

    matchers = ["lexical", "hybrid", "ann", "llm"]
    summary = {}
    failures = []

    for matcher in matchers:
        p1_list, p5_list = [], []
        r1_list, r5_list = [], []
        cov_list = []
        errors = Counter()

        for q, exp in tests:
            results = call_reconcile(q, matcher)

            p1_list.append(precision_at_k(results, exp, 1))
            p5_list.append(precision_at_k(results, exp, 5))
            r1_list.append(recall_at_k(results, exp, 1))
            r5_list.append(recall_at_k(results, exp, 5))
            cov_list.append(1 if results else 0)

            err = classify_error(q, exp, results)
            errors[err] += 1

            if err != "Correct":
                failures.append({
                    "query": q,
                    "expected": exp,
                    "top_ids": ";".join([r.get("id") for r in results]),
                    "top_names": ";".join([r.get("name") for r in results]),
                    "matcher": matcher,
                    "error_type": err
                })

        summary[matcher] = {
            "precision@1": mean(p1_list),
            "precision@5": mean(p5_list),
            "recall@1": mean(r1_list),
            "recall@5": mean(r5_list),
            "f1@1": f1(mean(p1_list), mean(r1_list)),
            "f1@5": f1(mean(p5_list), mean(r5_list)),
            "coverage": mean(cov_list),
            "errors": dict(errors)
        }

    # Save failures
    with open("evaluation_failures.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["query", "expected", "top_ids", "top_names", "matcher", "error_type"])
        writer.writeheader()
        for f in failures:
            writer.writerow(f)

    # Threshold sweep
    thresholds = [0.40, 0.50, 0.60, 0.70]
    sweep_results = {
        m: threshold_sweep(tests, m, thresholds)
        for m in matchers
    }

    with open("threshold_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["matcher", "threshold", "precision@1", "recall@1", "f1@1"])
        for m, rows in sweep_results.items():
            for th, P, R, F in rows:
                writer.writerow([m, th, P, R, F])

    # Runtime
    runtime = {m: measure_runtime(tests, m) for m in matchers}
    with open("runtime_stats.json", "w", encoding="utf-8") as fh:
        json.dump(runtime, fh, indent=2)

    # Per-country
    country_breakdown = {m: per_country_metrics(tests, m) for m in matchers}

    # Final summary
    final = {
        "summary": summary,
        "runtime": runtime,
        "country_breakdown": country_breakdown
    }

    with open("evaluation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(final, fh, indent=2)

    print("Evaluation complete. See output files:")
    print("- evaluation_summary.json")
    print("- evaluation_failures.csv")
    print("- threshold_sweep.csv")
    print("- runtime_stats.json")


if __name__ == "__main__":
    evaluate_all()

