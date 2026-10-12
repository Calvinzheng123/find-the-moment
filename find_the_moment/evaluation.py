"""User-labeled temporal retrieval evaluation; never fabricate ground truth."""

import csv
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path


def overlap(start, end, a, b):
    return max(0, min(end, b) - max(start, a))


def validate_cases(data, store):
    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation JSON must be a nonempty list of labeled queries.")
    for i, c in enumerate(data, 1):
        if not isinstance(c, dict) or not str(c.get("query", "")).strip():
            raise ValueError(f"Case {i} needs a query.")
        rid = c.get("recording_id")
        recording = store.get(rid)
        if not isinstance(c.get("present", True), bool):
            raise ValueError(f"Case {i}: present must be true or false.")
        if c.get("present", True):
            a, b = float(c["expected_start"]), float(c["expected_end"])
            if (
                not math.isfinite(a)
                or not math.isfinite(b)
                or not 0 <= a < b <= recording["duration"]
            ):
                raise ValueError(f"Case {i}: target interval is invalid.")
    return data


def score_hits(hits, case):
    if not case.get("present", True):
        # No calibrated absence detector exists. Report returned candidates, not "accuracy".
        return {
            "hit1": None,
            "hit3": None,
            "mrr3": None,
            "context_coverage": None,
            "returned_for_absent_query": len(hits),
        }
    a, b = float(case["expected_start"]), float(case["expected_end"])
    rid = case["recording_id"]

    # Hit if intersection covers >=25% of the shorter interval; short near-misses excluded.
    def correct(p):
        if p["recording_id"] != rid:
            return False
        intersection = overlap(p["start"], p["end"], a, b)
        return intersection / max(0.001, min(p["end"] - p["start"], b - a)) >= 0.25

    ranks = [i + 1 for i, p in enumerate(hits[:3]) if correct(p)]
    coverage = 0
    if hits and hits[0]["recording_id"] == rid:
        coverage = overlap(hits[0]["play_start"], hits[0]["play_end"], a, b) / (b - a)
    return {
        "hit1": int(bool(hits and correct(hits[0]))),
        "hit3": int(bool(ranks)),
        "mrr3": 1 / min(ranks) if ranks else 0,
        "context_coverage": coverage,
        "returned_for_absent_query": None,
    }


def run_evaluation(engine, cases, scope, methods, progress=lambda *a, **k: None):
    cases = validate_cases(cases, engine.store)
    records = []
    if not scope:
        raise ValueError("Select the recordings to evaluate in the Search tab.")
    if any(c["recording_id"] not in scope for c in cases):
        raise ValueError("Every labeled recording must be included in the selected search scope.")
    total = len(cases) * len(methods)
    for method in methods:
        # Warm embedding/reranker weights once; timing then measures warm queries.
        engine.search(cases[0]["query"], scope, "Spoken content", method, count=3)
        for c in cases:
            hits, seconds, warnings = engine.search(
                c["query"], scope, "Spoken content", method, count=3, strategy="Explanation"
            )
            scores = score_hits(hits, c)
            records.append(
                dict(
                    query=c["query"],
                    recording_id=c["recording_id"],
                    method=method,
                    search_seconds=seconds,
                    **scores,
                    warnings="; ".join(warnings),
                    returned=json.dumps(
                        [
                            {
                                "recording_id": p["recording_id"],
                                "start": p["start"],
                                "end": p["end"],
                                "play_start": p["play_start"],
                                "play_end": p["play_end"],
                            }
                            for p in hits
                        ]
                    ),
                )
            )
            progress(len(records) / total, desc=f"Evaluating {len(records)}/{total} queries…")
    summaries = []
    for method in methods:
        rows = [r for r in records if r["method"] == method]
        positive = [r for r in rows if r["hit1"] is not None]
        absent = [r for r in rows if r["hit1"] is None]
        summaries.append(
            {
                "method": method,
                "positive_queries": len(positive),
                "absent_queries": len(absent),
                "hit1": statistics.mean(r["hit1"] for r in positive) if positive else None,
                "hit3": statistics.mean(r["hit3"] for r in positive) if positive else None,
                "mrr3": statistics.mean(r["mrr3"] for r in positive) if positive else None,
                "mean_context_coverage": statistics.mean(r["context_coverage"] for r in positive)
                if positive
                else None,
                "median_search_seconds": statistics.median(r["search_seconds"] for r in rows),
                "absent_queries_returning_candidates": sum(
                    r["returned_for_absent_query"] > 0 for r in absent
                ),
            }
        )
    folder = engine.store.root / "evaluations"
    folder.mkdir(exist_ok=True)
    stamp = str(time.time_ns())
    csv_path = folder / f"results-{stamp}.csv"
    json_path = folder / f"summary-{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(
        json.dumps(
            {
                "summaries": summaries,
                "label_sources": sorted(set(str(c.get("source", "unspecified")) for c in cases)),
                "reviewed_labels": sum(bool(c.get("reviewed", False)) for c in cases),
                "total_labels": len(cases),
                "criterion": "Temporal overlap covers at least 25% of the shorter target/matched interval. "
                "Context coverage is target fraction covered by first playback clip. "
                "Warm queries only; excludes initial model load, ingestion and transcription. "
                "Absence cases report returned candidates, not calibrated absence accuracy.",
            },
            indent=2,
        )
    )
    return summaries, str(csv_path), str(json_path)


def run_saved_evaluation(engine, scope, methods, progress=lambda *a, **k: None, force=False):
    """Cache keyed by labels, retrieval settings, and active index generations."""
    if not scope:
        raise ValueError("Select recordings in the Search tab first.")
    if not methods:
        raise ValueError("Choose at least one evaluation method.")
    cases = engine.store.cases(scope)
    if not cases:
        raise ValueError(
            "No saved answer labels for these recordings. Add them in Evaluation, or import a label file once."
        )
    unprocessed = [r for r in scope if not engine.store.get(r)["active_index"]]
    if unprocessed:
        raise ValueError("Process every selected recording before evaluating.")
    fingerprint = {
        "cache_version": 1,
        "scope": sorted(scope),
        "methods": methods,
        "cases": cases,
        "indexes": [(r, engine.store.get(r)["active_index"]) for r in sorted(scope)],
        "title_model": os.getenv("TITLE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        "whisper_model": os.getenv("WHISPER_MODEL", "base"),
    }
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    directory = engine.store.root / "evaluations"
    directory.mkdir(exist_ok=True)
    cache = directory / f"cache-{key}.json"
    if cache.is_file() and not force:
        data = json.loads(cache.read_text())
        if Path(data["csv"]).is_file() and Path(data["summary"]).is_file():
            return data["summaries"], data["csv"], data["summary"], True
    summaries, csv_path, json_path = run_evaluation(engine, cases, scope, methods, progress)
    temporary = cache.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"summaries": summaries, "csv": csv_path, "summary": json_path})
    )
    os.replace(temporary, cache)
    return summaries, csv_path, json_path, False
