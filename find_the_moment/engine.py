"""Persistent media processing, retrieval, correction, and collection search."""

import logging
import threading
import time

import numpy as np

from .core import (
    chapter_for,
    context_range,
    diverse,
    edit_chapter,
    group_segments,
    merge_chapters,
    retrieve,
    split_chapter,
    split_topics,
    validate_segments,
)
from .models import Models
from .storage import Store

LOCK = threading.RLock()


class Engine:
    def __init__(self, store=None, models=None):
        self.store = store or Store()
        self.models = models or Models()

    def build_index(self, recording_id, segments, generated, progress):
        self.store.get(recording_id)
        passages = group_segments(segments, 40, True)
        units = group_segments(segments, 30, False)
        vectors = (
            self.models.encode([p["text"] for p in passages], "Document")
            if passages
            else np.empty((0, 768), np.float32)
        )
        unit_vectors = (
            self.models.encode([u["text"] for u in units], "Document")
            if units
            else np.empty((0, 768), np.float32)
        )
        chapters = split_topics(units, unit_vectors)
        progress(0.7, desc="Classifying sections and passages…")
        for p, label in zip(passages, self.models.classify([p["text"] for p in passages])):
            p.update(label)
        for c, label in zip(
            chapters,
            self.models.classify(
                [
                    " ".join(
                        s["text"]
                        for s in segments
                        if s["start"] < c["end"] and s["end"] > c["start"]
                    )
                    for c in chapters
                ]
            ),
        ):
            c.update(label)
        warnings = []
        if generated:
            for i, c in enumerate(chapters):
                text = " ".join(
                    s["text"] for s in segments if s["start"] < c["end"] and s["end"] > c["start"]
                )
                progress(
                    0.75 + 0.2 * (i + 1) / max(1, len(chapters)),
                    desc=f"Writing title {i + 1}/{len(chapters)}…",
                )
                try:
                    c.update(self.models.describe(text))
                except Exception as exc:
                    logging.warning("Metadata fallback: %s", exc)
                    warnings.append(
                        f"Section {i + 1}: used transcript-based title because generation failed."
                    )
        return passages, chapters, vectors, warnings

    def import_recording(self, upload, url, generated, progress):
        from .media import prepare_source, source_key

        source_key(upload, url)
        started = time.perf_counter()
        prepared = prepare_source(upload, url, progress)
        recording_id, reused = self.store.ingest(prepared)
        acquisition = time.perf_counter() - started
        existing = self.store.load_index(recording_id)
        if existing:
            return recording_id, "Reused saved audio, transcript, chapters, and embeddings."
        return recording_id, self.process(recording_id, generated, progress, acquisition)

    def process(self, recording_id, generated, progress, acquisition=0):
        recording = self.store.get(recording_id)
        started = time.perf_counter()
        with LOCK:
            segments = self.models.transcribe(recording["wav"], progress, recording["duration"])
            transcription = time.perf_counter() - started
            passages, chapters, vectors, warnings = self.build_index(
                recording_id, segments, generated, progress
            )
            self.store.save_index(
                recording_id,
                segments,
                passages,
                chapters,
                vectors,
                {
                    "acquisition_seconds": acquisition,
                    "transcription_seconds": transcription,
                    "indexing_seconds": time.perf_counter() - started - transcription,
                    "processing_seconds": time.perf_counter() - started + acquisition,
                },
            )
        progress(1, desc="Saved recording and indexes.")
        return f"Saved {len(segments)} transcript segments and {len(chapters)} chapters. " + (
            " ".join(warnings) if warnings else ""
        )

    def correct_transcript(self, recording_id, rows, generated, progress):
        recording = self.store.get(recording_id)
        segments = validate_segments(rows, recording["duration"])
        started = time.perf_counter()
        with LOCK:
            passages, chapters, vectors, warnings = self.build_index(
                recording_id, segments, generated, progress
            )
            self.store.save_index(
                recording_id,
                segments,
                passages,
                chapters,
                vectors,
                {"last_reindex_seconds": time.perf_counter() - started},
            )
        return (
            "Corrections saved and passages reindexed. Chapters were rebuilt; prior chapter edits are retained in the previous index generation. "
            + " ".join(warnings)
        )

    def sounds(self, recording_id, seconds, progress):
        cached = self.store.load_sounds(recording_id, seconds)
        if cached is not None:
            return cached
        import soundfile as sf

        recording = self.store.get(recording_id)
        spans = []
        batches = []
        with LOCK, sf.SoundFile(recording["wav"]) as audio:
            width = int(seconds * 16000)
            hop = max(1, width // 2)
            start = 0
            while start < len(audio):
                end = min(start + width, len(audio))
                spans.append({"start": start / 16000, "end": end / 16000, "text": ""})
                if end == len(audio):
                    break
                start += hop
            for offset in range(0, len(spans), 2):
                inputs = []
                for p in spans[offset : offset + 2]:
                    a, b = round(p["start"] * 16000), round(p["end"] * 16000)
                    audio.seek(a)
                    inputs.append({"audio": audio.read(b - a, dtype="float32")})
                batches.append(self.models.encode(inputs))
                progress(
                    0.1 + 0.8 * (offset + len(inputs)) / len(spans),
                    desc=f"Indexing sounds: {offset + len(inputs)}/{len(spans)}",
                )
        vectors = np.concatenate(batches)
        self.store.save_sounds(recording_id, seconds, spans, vectors)
        return spans, vectors

    def search(
        self,
        query,
        recording_ids,
        mode="Spoken content",
        method="Hybrid + reranker",
        seconds=6,
        count=5,
        strategy="Explanation",
        padding=3,
        progress=lambda *a, **k: None,
    ):
        if not query or not query.strip():
            raise ValueError("Enter a search question or sound description.")
        if not recording_ids:
            raise ValueError("Select at least one processed recording.")
        if mode not in {"Spoken content", "Sounds", "Both"}:
            raise ValueError("Invalid search mode.")
        if count < 1 or count > 20:
            raise ValueError("Result count must be between 1 and 20.")
        started = time.perf_counter()
        speech_items = []
        speech_vectors = []
        all_sounds = []
        indexes = {}
        warnings = []
        with LOCK:
            q = (
                np.zeros(768, dtype=np.float32)
                if method == "Keyword" and mode == "Spoken content"
                else self.models.encode(query, "SearchQuery")
            )
            for rid in recording_ids:
                index = self.store.load_index(rid)
                if index is None:
                    warnings.append(
                        f"{self.store.get(rid)['title']}: no transcript index; process it first."
                    )
                    continue
                indexes[rid] = index
                recording = index["recording"]
                if mode in {"Spoken content", "Both"}:
                    for p in index["passages"]:
                        chapter = chapter_for(p["start"], p["end"], index["chapters"])
                        speech_items.append(
                            dict(
                                p,
                                recording_id=rid,
                                title=recording["title"],
                                mode="Speech",
                                chapter_title=chapter["title"] if chapter else "",
                                section_type=chapter["section_type"]
                                if chapter and chapter.get("reviewed")
                                else p.get("section_type", "Discussion"),
                            )
                        )
                    speech_vectors.append(index["vectors"])
                if mode in {"Sounds", "Both"}:
                    spans, vectors = self.sounds(rid, int(seconds), progress)
                    ranked = retrieve(
                        query,
                        [
                            dict(
                                p,
                                recording_id=rid,
                                title=recording["title"],
                                mode="Sound",
                                section_type="Sound",
                            )
                            for p in spans
                        ],
                        vectors,
                        q,
                        "Semantic",
                        20,
                    )
                    all_sounds += ranked
            speech = []
            if speech_items:
                corpus = np.concatenate(speech_vectors)
                speech = retrieve(query, speech_items, corpus, q, method, 20)
                if method == "Hybrid + reranker":
                    progress(0.9, desc="Reranking candidate passages…")
                    try:
                        values = self.models.rerank(query, speech)
                        for p, value in zip(speech, values):
                            p["rerank_score"] = float(value)
                        # Preserve a small explicit section-intent signal after reranking.
                        from .core import intent

                        target = intent(query)
                        speech.sort(
                            key=lambda p: (
                                p["rerank_score"]
                                + (0.4 if target and p["section_type"] == target else 0)
                            ),
                            reverse=True,
                        )
                    except Exception:
                        logging.exception("Reranker unavailable")
                        warnings.append("Reranker unavailable; returned hybrid retrieval results.")
            all_sounds.sort(key=lambda p: p["similarity"], reverse=True)
            if mode == "Both":
                combined = []
                for i in range(max(len(speech), len(all_sounds))):
                    if i < len(speech):
                        combined.append(speech[i])
                    if i < len(all_sounds):
                        combined.append(all_sounds[i])
            else:
                combined = speech if mode == "Spoken content" else all_sounds
            selected = diverse(combined, int(count))
        for p in selected:
            index = indexes[p["recording_id"]]
            r = index["recording"]
            p["play_start"], p["play_end"] = context_range(
                p,
                index["segments"],
                index["chapters"],
                r["duration"],
                strategy if p["mode"] == "Speech" else "Matched passage",
                padding,
            )
            p["video_id"] = r["video_id"]
        elapsed = time.perf_counter() - started
        progress(1, desc="Search complete.")
        return selected, elapsed, warnings

    def edit(self, recording_id, chapter_id, title, label, start, end):
        with LOCK:
            index = self.store.load_index(recording_id)
            chapters = edit_chapter(
                index["chapters"],
                chapter_id,
                title,
                label,
                start,
                end,
                index["recording"]["duration"],
            )
            self.store.save_chapters(recording_id, chapters)

    def split(self, recording_id, chapter_id, point):
        with LOCK:
            index = self.store.load_index(recording_id)
            chapters = split_chapter(
                index["chapters"],
                chapter_id,
                point,
                index["segments"],
                index["recording"]["duration"],
            )
            self.store.save_chapters(recording_id, chapters)

    def merge(self, recording_id, chapter_id):
        with LOCK:
            index = self.store.load_index(recording_id)
            chapters = merge_chapters(index["chapters"], chapter_id, index["recording"]["duration"])
            self.store.save_chapters(recording_id, chapters)
