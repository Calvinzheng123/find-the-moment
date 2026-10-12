"""Single-owner persistent SQLite catalog with immutable index generations."""

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np


class Store:
    def __init__(self, root=None):
        self.root = Path(root or os.getenv("MEDIA_LIBRARY_DIR", "./recording_library")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "catalog.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS recordings(
                  id TEXT PRIMARY KEY, title TEXT NOT NULL, directory TEXT NOT NULL,
                  duration REAL NOT NULL, video_id TEXT, active_index TEXT,
                  timings TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS bookmarks(
                  id TEXT PRIMARY KEY, recording_id TEXT NOT NULL,
                  start REAL NOT NULL, end REAL NOT NULL, note TEXT NOT NULL,
                  created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS sound_indexes(
                  recording_id TEXT NOT NULL, seconds INTEGER NOT NULL,
                  directory TEXT NOT NULL, PRIMARY KEY(recording_id,seconds));
                CREATE TABLE IF NOT EXISTS evaluation_cases(
                  id TEXT PRIMARY KEY, recording_id TEXT NOT NULL, query TEXT NOT NULL,
                  present INTEGER NOT NULL, expected_start REAL, expected_end REAL,
                  source TEXT NOT NULL, reviewed INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluation_initialized(recording_id TEXT PRIMARY KEY);
            """)

    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def list(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM recordings ORDER BY created DESC")]

    def get(self, recording_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM recordings WHERE id=?", (recording_id,)).fetchone()
        if row is None:
            raise ValueError("Recording not found. Refresh the library.")
        result = dict(row)
        result["timings"] = json.loads(result["timings"])
        result["wav"] = str(Path(result["directory"]) / "recording.wav")
        if not Path(result["wav"]).is_file():
            raise ValueError("Saved audio is missing. Import the recording again.")
        return result

    def ingest(self, prepared):
        digest = hashlib.sha256()
        with open(prepared["wav"], "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        recording_id = digest.hexdigest()[:24]
        with self.connect() as db:
            exists = db.execute("SELECT id FROM recordings WHERE id=?", (recording_id,)).fetchone()
        if exists:
            shutil.rmtree(prepared["directory"], ignore_errors=True)
            return recording_id, True
        destination = self.root / recording_id
        destination.mkdir(exist_ok=True)
        shutil.move(prepared["wav"], destination / "recording.wav")
        shutil.rmtree(prepared["directory"], ignore_errors=True)
        with self.connect() as db:
            db.execute(
                "INSERT INTO recordings VALUES(?,?,?,?,?,?,?,?)",
                (
                    recording_id,
                    prepared["title"],
                    str(destination),
                    prepared["duration"],
                    prepared.get("video_id"),
                    None,
                    "{}",
                    time.time(),
                ),
            )
        return recording_id, False

    def save_index(self, recording_id, segments, passages, chapters, vectors, timings=None):
        recording = self.get(recording_id)
        # Write a complete new generation before changing the active pointer.
        folder = Path(recording["directory"]) / f"index-{uuid.uuid4().hex}"
        folder.mkdir()
        data = {"segments": segments, "passages": passages, "chapters": chapters}
        try:
            (folder / "index.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            np.save(
                folder / "embeddings.npy", np.asarray(vectors, dtype=np.float32), allow_pickle=False
            )
            (folder / "transcript.txt").write_text(
                "\n".join(f"[{s['start']:.3f}–{s['end']:.3f}] {s['text']}" for s in segments),
                encoding="utf-8",
            )
            (folder / "chapters.json").write_text(
                json.dumps(chapters, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            values = dict(recording["timings"])
            values.update(timings or {})
            with self.connect() as db:
                db.execute(
                    "UPDATE recordings SET active_index=?,timings=? WHERE id=?",
                    (str(folder), json.dumps(values), recording_id),
                )
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def load_index(self, recording_id):
        recording = self.get(recording_id)
        if not recording["active_index"]:
            return None
        directory = Path(recording["active_index"])
        data = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        data["vectors"] = np.load(directory / "embeddings.npy", allow_pickle=False)
        data["recording"] = recording
        data["transcript_path"] = str(directory / "transcript.txt")
        data["chapter_path"] = str(directory / "chapters.json")
        return data

    def save_chapters(self, recording_id, chapters):
        index = self.load_index(recording_id)
        if index is None:
            raise ValueError("Process the recording first.")
        self.save_index(
            recording_id, index["segments"], index["passages"], chapters, index["vectors"]
        )

    def save_sounds(self, recording_id, seconds, spans, vectors):
        recording = self.get(recording_id)
        folder = Path(recording["directory"]) / f"sound-{seconds}-{uuid.uuid4().hex}"
        folder.mkdir()
        (folder / "spans.json").write_text(json.dumps(spans), encoding="utf-8")
        np.save(
            folder / "embeddings.npy", np.asarray(vectors, dtype=np.float32), allow_pickle=False
        )
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO sound_indexes VALUES(?,?,?)",
                (recording_id, seconds, str(folder)),
            )

    def load_sounds(self, recording_id, seconds):
        with self.connect() as db:
            row = db.execute(
                "SELECT directory FROM sound_indexes WHERE recording_id=? AND seconds=?",
                (recording_id, seconds),
            ).fetchone()
        if not row:
            return None
        folder = Path(row["directory"])
        return json.loads((folder / "spans.json").read_text()), np.load(
            folder / "embeddings.npy", allow_pickle=False
        )

    def bookmark(self, recording_id, start, end, note):
        recording = self.get(recording_id)
        if not 0 <= start < end <= recording["duration"]:
            raise ValueError("Bookmark must be inside the recording, with end after start.")
        with self.connect() as db:
            db.execute(
                "INSERT INTO bookmarks VALUES(?,?,?,?,?,?)",
                (uuid.uuid4().hex, recording_id, float(start), float(end), str(note), time.time()),
            )

    def bookmarks(self):
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute("""SELECT b.*,r.title FROM bookmarks b
                JOIN recordings r ON b.recording_id=r.id ORDER BY b.created DESC""")
            ]

    def delete_bookmark(self, bookmark_id):
        with self.connect() as db:
            db.execute("DELETE FROM bookmarks WHERE id=?", (bookmark_id,))

    def export_bookmarks(self):
        path = self.root / "bookmarks.json"
        path.write_text(
            json.dumps(self.bookmarks(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)

    def cases(self, recording_ids):
        if not recording_ids:
            return []
        placeholders = ",".join("?" for _ in recording_ids)
        with self.connect() as db:
            rows = db.execute(
                f"SELECT * FROM evaluation_cases WHERE recording_id IN ({placeholders}) ORDER BY recording_id,rowid",
                tuple(recording_ids),
            ).fetchall()
        return [
            dict(
                recording_id=r["recording_id"],
                query=r["query"],
                present=bool(r["present"]),
                **(
                    {"expected_start": r["expected_start"], "expected_end": r["expected_end"]}
                    if r["present"]
                    else {}
                ),
                source=r["source"],
                reviewed=bool(r["reviewed"]),
            )
            for r in rows
        ]

    def labels_initialized(self, recording_id):
        with self.connect() as db:
            return bool(
                db.execute(
                    "SELECT 1 FROM evaluation_initialized WHERE recording_id=?", (recording_id,)
                ).fetchone()
            )

    def save_cases(self, recording_id, cases):
        from .evaluation import validate_cases

        if cases:
            validate_cases(cases, self)
            if any(c["recording_id"] != recording_id for c in cases):
                raise ValueError("Labels must belong to the selected recording.")
        else:
            self.get(recording_id)
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO evaluation_initialized VALUES(?)", (recording_id,))
            db.execute("DELETE FROM evaluation_cases WHERE recording_id=?", (recording_id,))
            for c in cases:
                db.execute(
                    "INSERT INTO evaluation_cases VALUES(?,?,?,?,?,?,?,?)",
                    (
                        uuid.uuid4().hex,
                        recording_id,
                        c["query"],
                        int(c.get("present", True)),
                        c.get("expected_start") if c.get("present", True) else None,
                        c.get("expected_end") if c.get("present", True) else None,
                        str(c.get("source", "manual")),
                        int(c.get("reviewed", False)),
                    ),
                )

    def backup(self):
        """Portable consistent snapshot, including audio and index generations."""
        target = Path(tempfile.gettempdir()) / f"moment-library-{uuid.uuid4().hex}.zip"
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "catalog.sqlite3"
            with self.connect() as source, sqlite3.connect(database) as output:
                source.backup(output)
                output.execute(
                    "UPDATE recordings SET directory=replace(directory,?,?),active_index=replace(active_index,?,?)",
                    (str(self.root), "LIBRARY_ROOT", str(self.root), "LIBRARY_ROOT"),
                )
                output.execute(
                    "UPDATE sound_indexes SET directory=replace(directory,?,?)",
                    (str(self.root), "LIBRARY_ROOT"),
                )
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(database, "catalog.sqlite3")
                for path in self.root.rglob("*"):
                    if (
                        path.is_file()
                        and path != self.db
                        and not path.name.startswith("catalog.sqlite3-")
                    ):
                        archive.write(path, path.relative_to(self.root))
        return str(target)

    def restore(self, archive_path):
        if self.list():
            raise ValueError(
                "Restore requires an empty library, so existing recordings are not overwritten."
            )
        with tempfile.TemporaryDirectory() as temporary:
            with zipfile.ZipFile(archive_path) as archive:
                if sum(info.file_size for info in archive.infolist()) > 5 * 1024**3:
                    raise ValueError("Backup is larger than the 5 GB restore limit.")
                for info in archive.infolist():
                    path = Path(info.filename)
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or stat.S_ISLNK(info.external_attr >> 16)
                    ):
                        raise ValueError("Unsafe path in backup archive.")
                archive.extractall(temporary)
            source = Path(temporary)
            if not (source / "catalog.sqlite3").is_file():
                raise ValueError("This is not a Find the Moment library backup.")
            # Validate before replacing the empty database.
            with sqlite3.connect(source / "catalog.sqlite3") as db:
                db.execute("SELECT id,directory,active_index FROM recordings").fetchall()
                db.execute("SELECT directory FROM sound_indexes").fetchall()
                db.execute("SELECT recording_id,start,end,note FROM bookmarks").fetchall()
                # Backups made before label caching are upgraded after restore.
                db.execute(
                    "CREATE TABLE IF NOT EXISTS evaluation_cases(id TEXT PRIMARY KEY,recording_id TEXT NOT NULL,query TEXT NOT NULL,present INTEGER NOT NULL,expected_start REAL,expected_end REAL,source TEXT NOT NULL,reviewed INTEGER NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS evaluation_initialized(recording_id TEXT PRIMARY KEY)"
                )
                db.execute(
                    "INSERT OR IGNORE INTO evaluation_initialized SELECT DISTINCT recording_id FROM evaluation_cases"
                )
                db.execute(
                    "UPDATE recordings SET directory=replace(directory,?,?),active_index=replace(active_index,?,?)",
                    ("LIBRARY_ROOT", str(self.root), "LIBRARY_ROOT", str(self.root)),
                )
                db.execute(
                    "UPDATE sound_indexes SET directory=replace(directory,?,?)",
                    ("LIBRARY_ROOT", str(self.root)),
                )
                for row in db.execute("SELECT directory,active_index FROM recordings"):
                    for stored_path in row:
                        if stored_path is not None and not Path(
                            stored_path
                        ).resolve().is_relative_to(self.root):
                            raise ValueError("Backup contains an external recording path.")
                for (stored_path,) in db.execute("SELECT directory FROM sound_indexes"):
                    if not Path(stored_path).resolve().is_relative_to(self.root):
                        raise ValueError("Backup contains an external sound-index path.")
            for path in source.iterdir():
                if path.name == "catalog.sqlite3":
                    continue
                shutil.move(str(path), str(self.root / path.name))
            os.replace(source / "catalog.sqlite3", self.db)
