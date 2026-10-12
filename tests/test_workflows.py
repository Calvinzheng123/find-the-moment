"""Offline workflow tests with deterministic fake model outputs."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from find_the_moment.core import (
    context_range,
)
from find_the_moment.engine import Engine
from find_the_moment.evaluation import run_evaluation, run_saved_evaluation, score_hits
from find_the_moment.storage import Store
from find_the_moment.ui_assets import timeline_html, transcript_html


class FakeModels:
    def encode(self, items, prompt=None):
        if isinstance(items, str):
            return self.vector(items)
        return np.array([self.vector(x) for x in items], dtype=np.float32)

    def vector(self, text):
        text = str(text).lower()
        v = np.zeros(768, dtype=np.float32)
        v[0] = 1 if any(t in text for t in ["dropout", "overfit"]) else 0
        v[1] = 1 if "example" in text else 0
        v[2] = 1 if "definition" in text else 0
        v[3] = 1 if "gradient" in text else 0
        if not v.any():
            v[4] = 1
        return v / np.linalg.norm(v)

    def classify(self, texts):
        return [
            {
                "section_type": "Worked example" if "example" in t.lower() else "Definition",
                "label_source": "fake test",
            }
            for t in texts
        ]

    def describe(self, text):
        return {
            "title": "Readable test title",
            "summary": "Test summary",
            "metadata_source": "test fake",
        }

    def rerank(self, query, passages):
        return np.array([10 if "example" in p["text"].lower() else 0 for p in passages])


class Workflows(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "library")
        self.engine = Engine(self.store, FakeModels())
        self.ids = []
        for i in range(2):
            folder = self.root / f"source{i}"
            folder.mkdir()
            (folder / "recording.wav").write_bytes(b"waveform" + bytes([i]))
            rid, _ = self.store.ingest(
                {
                    "directory": str(folder),
                    "wav": str(folder / "recording.wav"),
                    "title": f"Lecture {i}",
                    "duration": 200,
                    "video_id": None,
                }
            )
            self.ids.append(rid)
        self.segments = [
            {
                "start": i * 20,
                "end": (i + 1) * 20,
                "text": (
                    "Dropout definition prevents overfitting."
                    if i < 5
                    else "For example dropout changes this model."
                ),
            }
            for i in range(10)
        ]
        self.build(self.ids[0], self.segments)
        self.build(
            self.ids[1],
            [dict(s, text="Gradient definition and gradient example.") for s in self.segments],
        )

    def tearDown(self):
        self.temp.cleanup()

    def build(self, rid, segments):
        p, c, v, w = self.engine.build_index(rid, segments, True, lambda *a, **k: None)
        self.store.save_index(rid, segments, p, c, v)

    def test_saved_library_survives_new_store(self):
        again = Store(self.store.root)
        self.assertEqual(len(again.list()), 2)
        index = again.load_index(self.ids[0])
        self.assertEqual(index["chapters"][0]["title"], "Readable test title")
        self.assertEqual(index["vectors"].shape[1], 768)

    def test_audio_deduplication(self):
        folder = self.root / "duplicate"
        folder.mkdir()
        (folder / "recording.wav").write_bytes(b"waveform\x00")
        rid, reused = self.store.ingest(
            {
                "directory": str(folder),
                "wav": str(folder / "recording.wav"),
                "title": "duplicate",
                "duration": 200,
            }
        )
        self.assertTrue(reused)
        self.assertEqual(rid, self.ids[0])
        self.assertFalse(folder.exists())

    def test_cross_recording_reranking(self):
        hits, seconds, warnings = self.engine.search("worked example dropout", self.ids, count=3)
        self.assertEqual(hits[0]["recording_id"], self.ids[0])
        self.assertIn("example", hits[0]["text"].lower())
        self.assertIn("rerank_score", hits[0])
        self.assertGreaterEqual(seconds, 0)

    def test_keyword_baseline(self):
        hits, _, _ = self.engine.search("gradient", self.ids, method="Keyword")
        self.assertTrue(hits)
        self.assertTrue(all(p["recording_id"] == self.ids[1] for p in hits))
        self.assertEqual(self.engine.search("absent-term", self.ids, method="Keyword")[0], [])

    def test_reranker_fallback_is_visible(self):
        self.engine.models.rerank = lambda *args: (_ for _ in ()).throw(RuntimeError("offline"))
        hits, _, warnings = self.engine.search("dropout", self.ids)
        self.assertTrue(hits)
        self.assertTrue(any("unavailable" in w for w in warnings))

    def test_corrections_reindex_and_keep_prior_generation(self):
        original = self.store.get(self.ids[0])["active_index"]
        rows = [
            [s["start"], s["end"], "Corrected technical term zirconium."] for s in self.segments
        ]
        self.engine.correct_transcript(self.ids[0], rows, False, lambda *a, **k: None)
        self.assertTrue(Path(original).is_dir())
        hits, _, _ = self.engine.search("zirconium", self.ids, method="Keyword")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["recording_id"], self.ids[0])
        self.assertTrue(
            all("zirconium" in s["text"] for s in self.store.load_index(self.ids[0])["segments"])
        )

    def test_chapter_edit_split_merge(self):
        rid = self.ids[0]
        index = self.store.load_index(rid)
        chapter = index["chapters"][0]
        self.engine.edit(rid, 0, "My title", "Worked example", chapter["start"], chapter["end"])
        updated = self.store.load_index(rid)["chapters"]
        self.assertEqual(updated[0]["title"], "My title")
        self.engine.split(rid, 0, 40)
        split = self.store.load_index(rid)["chapters"]
        self.assertEqual(split[0]["end"], 40)
        self.engine.merge(rid, 0)
        merged = self.store.load_index(rid)["chapters"]
        self.assertEqual(merged[0]["end"], chapter["end"])

    def test_invalid_transcript_does_not_replace_index(self):
        original = self.store.get(self.ids[0])["active_index"]
        with self.assertRaises(ValueError):
            self.engine.correct_transcript(
                self.ids[0], [[20, 10, "bad"]], False, lambda *a, **k: None
            )
        self.assertEqual(original, self.store.get(self.ids[0])["active_index"])

    def test_bookmarks_persist_export_delete(self):
        self.store.bookmark(self.ids[0], 10, 30, "Review example")
        marks = Store(self.store.root).bookmarks()
        self.assertEqual(marks[0]["note"], "Review example")
        self.assertEqual(len(json.loads(Path(self.store.export_bookmarks()).read_text())), 1)
        self.store.delete_bookmark(marks[0]["id"])
        self.assertEqual(self.store.bookmarks(), [])
        with self.assertRaises(ValueError):
            self.store.bookmark(self.ids[0], 190, 210, "bad")

    def test_portable_library_backup_restore(self):
        self.store.bookmark(self.ids[0], 10, 30, "Restored note")
        archive = self.store.backup()
        self.addCleanup(lambda: Path(archive).unlink(missing_ok=True))
        restored = Store(self.root / "restored")
        restored.restore(archive)
        self.assertEqual(len(restored.list()), 2)
        self.assertEqual(restored.bookmarks()[0]["note"], "Restored note")
        self.assertEqual(
            restored.load_index(self.ids[0])["chapters"][0]["title"], "Readable test title"
        )
        self.assertTrue(Path(restored.get(self.ids[0])["wav"]).is_file())
        with self.assertRaises(ValueError):
            restored.restore(archive)

    def test_context_range_and_html_safety(self):
        hit = {"start": 110, "end": 130}
        chapters = self.store.load_index(self.ids[0])["chapters"]
        start, end = context_range(hit, self.segments, chapters, 200, "Explanation", 3)
        self.assertLessEqual(start, 110)
        self.assertGreaterEqual(end, 130)
        self.assertLessEqual(end, 200)
        rendered = transcript_html([{"start": 0, "end": 1, "text": "<script>alert(1)</script>"}])
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("data-recording-seek", timeline_html(chapters, 200))

    def test_temporal_eval_and_absence_reporting(self):
        case = {"recording_id": self.ids[0], "expected_start": 100, "expected_end": 120}
        hit = {
            "recording_id": self.ids[0],
            "start": 100,
            "end": 130,
            "play_start": 90,
            "play_end": 140,
        }
        scores = score_hits([hit], case)
        self.assertEqual(scores["hit1"], 1)
        self.assertEqual(scores["context_coverage"], 1)
        self.assertIsNone(score_hits([hit], dict(case, present=False))["hit1"])
        cases = [
            dict(case, query="dropout example", present=True),
            {"recording_id": self.ids[0], "query": "absent term", "present": False},
        ]
        summaries, csv_file, json_file = run_evaluation(
            self.engine, cases, self.ids, ["Keyword", "Hybrid"]
        )
        self.assertEqual(len(summaries), 2)
        self.assertTrue(Path(csv_file).is_file())
        self.assertTrue(Path(json_file).is_file())
        self.assertEqual(summaries[0]["positive_queries"], 1)
        self.assertEqual(summaries[0]["absent_queries"], 1)

    def test_saved_evaluation_cache_and_invalidation(self):
        rid = self.ids[0]
        cases = [
            dict(
                recording_id=rid,
                query="dropout example",
                present=True,
                expected_start=100,
                expected_end=120,
                source="manual",
                reviewed=True,
            )
        ]
        self.store.save_cases(rid, cases)
        self.assertTrue(Store(self.store.root).labels_initialized(rid))
        first = run_saved_evaluation(self.engine, [rid], ["Keyword"])
        self.assertFalse(first[-1])
        second = run_saved_evaluation(self.engine, [rid], ["Keyword"])
        self.assertTrue(second[-1])
        self.assertEqual(first[1], second[1])
        original = self.engine.search
        self.engine.search = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("cache should bypass retrieval")
        )
        self.assertTrue(run_saved_evaluation(self.engine, [rid], ["Keyword"])[-1])
        self.engine.search = original
        self.assertFalse(run_saved_evaluation(self.engine, [rid], ["Keyword"], force=True)[-1])
        cases[0]["expected_end"] = 130
        self.store.save_cases(rid, cases)
        self.assertFalse(run_saved_evaluation(self.engine, [rid], ["Keyword"])[-1])
        self.engine.edit(
            rid,
            0,
            "Edited title",
            "Definition",
            0,
            self.store.load_index(rid)["chapters"][0]["end"],
        )
        self.assertFalse(run_saved_evaluation(self.engine, [rid], ["Keyword"])[-1])
        self.store.save_cases(rid, [])
        self.assertTrue(self.store.labels_initialized(rid))
        with self.assertRaises(ValueError):
            run_saved_evaluation(self.engine, [rid], ["Keyword"])


if __name__ == "__main__":
    unittest.main()
