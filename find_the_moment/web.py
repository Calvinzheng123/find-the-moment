"""Gradio interface and callbacks."""

import html
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlencode

import gradio as gr
import numpy as np
import soundfile as sf

from .core import LABELS
from .engine import Engine
from .evaluation import run_saved_evaluation, validate_cases
from .ui_assets import CSS, JS, timeline_html, transcript_html

ENGINE = Engine()


def require_recording(recording_id):
    if not recording_id:
        raise ValueError("Select a saved recording first.")
    return recording_id


def friendly(exc):
    logging.exception("Operation failed")
    message = str(exc)
    if "No space left" in message:
        message = "Storage is full. Use a larger persistent disk or a shorter recording."
    if "expired" in message.lower():
        message = "Session expired. Refresh and select the recording from the saved library."
    return gr.Error(message[:700])


def choices():
    return [(r["title"] + " · " + r["id"][:8], r["id"]) for r in ENGINE.store.list()]


def seed_known_labels():
    preset = Path(__file__).parent / "data" / "mit_lecture01.json"
    if not preset.is_file():
        return
    for recording in ENGINE.store.list():
        if (
            "mit6_7960f24_lec01" in recording["title"].lower()
            and abs(recording["duration"] - 3651) <= 3
            and not ENGINE.store.labels_initialized(recording["id"])
        ):
            cases = json.loads(preset.read_text())
            for c in cases:
                c["recording_id"] = recording["id"]
            ENGINE.store.save_cases(recording["id"], cases)


def blank_view():
    return (
        [],
        gr.update(choices=[], value=None),
        [],
        None,
        None,
        None,
        "",
        "",
        "No recording selected.",
    )


def view(recording_id):
    if not recording_id:
        return blank_view()
    try:
        index = ENGINE.store.load_index(recording_id)
        recording = ENGINE.store.get(recording_id)
        if index is None:
            return (
                [],
                gr.update(choices=[], value=None),
                [],
                None,
                None,
                recording["wav"],
                "",
                "",
                "Imported audio has no transcript index; use Process selected recording.",
            )
        chapters = index["chapters"]
        segments = index["segments"]
        rows = [
            [
                c["id"] + 1,
                round(c["start"], 2),
                round(c["end"], 2),
                c["title"],
                c["section_type"],
                c["summary"],
                c.get("metadata_source", ""),
            ]
            for c in chapters
        ]
        options = [(f"{c['id'] + 1}. {c['title']}", str(c["id"])) for c in chapters]
        timings = recording["timings"]
        stats = (
            f"{recording['title']} · {recording['duration'] / 60:.1f} minutes · "
            f"{len(segments)} transcript segments · {len(chapters)} sections. "
            f"Processing: {timings.get('processing_seconds', 0):.1f}s; "
            f"transcription: {timings.get('transcription_seconds', 0):.1f}s; indexing: {timings.get('indexing_seconds', 0):.1f}s."
        )
        return (
            rows,
            gr.update(choices=options, value=None),
            [[s["start"], s["end"], s["text"]] for s in segments],
            index["transcript_path"],
            index["chapter_path"],
            recording["wav"],
            transcript_html(segments),
            timeline_html(chapters, recording["duration"]),
            stats,
        )
    except Exception as exc:
        raise friendly(exc) from exc


def refresh():
    seed_known_labels()
    options = choices()
    ids = [v for _, v in options]
    return (
        gr.update(choices=options, value=ids[0] if ids else None),
        gr.update(choices=options, value=ids),
        *view(ids[0] if ids else None),
    )


def import_media(upload, url, generated, progress=gr.Progress()):
    try:
        rid, message = ENGINE.import_recording(upload, url or "", generated, progress)
        seed_known_labels()
        options = choices()
        return (
            gr.update(choices=options, value=rid),
            gr.update(choices=options, value=[v for _, v in options]),
            *view(rid),
            message,
        )
    except Exception as exc:
        raise friendly(exc) from exc


def process_selected(rid, generated, progress=gr.Progress()):
    try:
        message = ENGINE.process(require_recording(rid), generated, progress)
        return *view(rid), message
    except Exception as exc:
        raise friendly(exc) from exc


def correct(rid, rows, generated, progress=gr.Progress()):
    try:
        if hasattr(rows, "values"):
            rows = rows.values.tolist()
        message = ENGINE.correct_transcript(require_recording(rid), rows, generated, progress)
        return *view(rid), message
    except Exception as exc:
        raise friendly(exc) from exc


def chapter_fields(rid, cid):
    if not rid or cid is None:
        return "", "Discussion", 0, 0
    c = ENGINE.store.load_index(rid)["chapters"][int(cid)]
    return c["title"], c["section_type"], c["start"], c["end"]


def chapter_edit(rid, cid, title, label, start, end):
    try:
        if cid is None:
            raise ValueError("Choose a chapter first.")
        ENGINE.edit(require_recording(rid), int(cid), title, label, start, end)
        return *view(rid), "Chapter changes saved."
    except Exception as exc:
        raise friendly(exc) from exc


def chapter_split(rid, cid, point):
    try:
        if cid is None:
            raise ValueError("Choose a chapter first.")
        ENGINE.split(require_recording(rid), int(cid), point)
        return *view(rid), "Chapter split at the nearest transcript boundary."
    except Exception as exc:
        raise friendly(exc) from exc


def chapter_merge(rid, cid):
    try:
        if cid is None:
            raise ValueError("Choose a chapter first.")
        ENGINE.merge(require_recording(rid), int(cid))
        return *view(rid), "Chapter merged with the following chapter."
    except Exception as exc:
        raise friendly(exc) from exc


def audio_clip(recording_id, start, end):
    recording = ENGINE.store.get(recording_id)
    with sf.SoundFile(recording["wav"]) as audio:
        a, b = round(start * 16000), round(end * 16000)
        audio.seek(a)
        samples = audio.read(b - a, dtype="float32")
    return (16000, (np.clip(samples, -1, 1) * 32767).astype(np.int16))


def search(query, scope, mode, method, seconds, count, strategy, padding, progress=gr.Progress()):
    try:
        hits, elapsed, warnings = ENGINE.search(
            query,
            scope or [],
            mode,
            method,
            int(seconds),
            int(count),
            strategy,
            int(padding),
            progress,
        )
        rows = [
            [
                i + 1,
                p["title"],
                p["mode"],
                p.get("section_type", ""),
                round(p["start"], 2),
                round(p["end"], 2),
                round(p["play_start"], 2),
                round(p["play_end"], 2),
                round(p["similarity"], 4),
                round(p["rerank_score"], 3) if "rerank_score" in p else None,
                p["text"],
            ]
            for i, p in enumerate(hits)
        ]
        options = [
            (f"{i + 1}. {p['title']} · {p['start']:.1f}s", str(i)) for i, p in enumerate(hits)
        ]
        message = (
            f"{len(hits)} candidates in {elapsed:.2f}s. Scores are not confidence probabilities. "
            + " ".join(warnings)
        )
        return hits, rows, gr.update(choices=options, value=None), message
    except Exception as exc:
        raise friendly(exc) from exc


def open_result(result_id, hits):
    if result_id is None or not hits:
        return None, "", "", None, "", "", None, 0, 0
    p = hits[int(result_id)]
    index = ENGINE.store.load_index(p["recording_id"])
    recording = index["recording"]
    description = (
        f"<h3>{html.escape(p['title'])}</h3>"
        f"<p><b>Matched passage:</b> {p['start']:.1f}–{p['end']:.1f}s · "
        f"<b>Expanded playback:</b> {p['play_start']:.1f}–{p['play_end']:.1f}s</p>"
        f"<p>{html.escape(p['text'])}</p>"
    )
    if p.get("video_id"):
        url = "https://www.youtube.com/watch?" + urlencode(
            {"v": p["video_id"], "t": int(p["start"])}
        )
        description += f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">Open original video at match</a>'
    same = [r for r in hits if r["recording_id"] == p["recording_id"]]
    offset = f'<span id="playback-offset" data-offset="{p["play_start"]}"></span>'
    return (
        audio_clip(p["recording_id"], p["play_start"], p["play_end"]),
        description,
        offset,
        recording["wav"],
        transcript_html(index["segments"], p),
        timeline_html(index["chapters"], recording["duration"], same),
        p["recording_id"],
        p["play_start"],
        p["play_end"],
    )


def play_section(rid, cid):
    try:
        if cid is None:
            raise ValueError("Choose a chapter first.")
        index = ENGINE.store.load_index(require_recording(rid))
        c = index["chapters"][int(cid)]
        return (
            audio_clip(rid, c["start"], c["end"]),
            f"<h3>{html.escape(c['title'])}</h3><p>{html.escape(c['summary'])}</p>",
            f'<span id="playback-offset" data-offset="{c["start"]}"></span>',
            index["recording"]["wav"],
            transcript_html(index["segments"]),
            timeline_html(index["chapters"], index["recording"]["duration"]),
            rid,
            c["start"],
            c["end"],
        )
    except Exception as exc:
        raise friendly(exc) from exc


def bookmarks_view():
    marks = ENGINE.store.bookmarks()
    return (
        [[m["title"], m["start"], m["end"], m["note"]] for m in marks],
        gr.update(
            choices=[
                (f"{m['title']} · {m['start']:.1f}s · {m['note'][:40]}", m["id"]) for m in marks
            ],
            value=None,
        ),
        ENGINE.store.export_bookmarks(),
    )


def save_bookmark(rid, start, end, note):
    try:
        ENGINE.store.bookmark(require_recording(rid), float(start), float(end), note or "")
        return *bookmarks_view(), "Bookmark saved."
    except Exception as exc:
        raise friendly(exc) from exc


def delete_bookmark(mark_id):
    if not mark_id:
        raise gr.Error("Choose a bookmark to delete.")
    ENGINE.store.delete_bookmark(mark_id)
    return *bookmarks_view(), "Bookmark deleted."


def play_bookmark(mark_id):
    try:
        m = next((m for m in ENGINE.store.bookmarks() if m["id"] == mark_id), None)
        if not m:
            raise ValueError("Choose a saved bookmark.")
        index = ENGINE.store.load_index(m["recording_id"])
        return (
            audio_clip(m["recording_id"], m["start"], m["end"]),
            "<p>" + html.escape(m["note"]) + "</p>",
            f'<span id="playback-offset" data-offset="{m["start"]}"></span>',
            index["recording"]["wav"],
            transcript_html(index["segments"]),
            timeline_html(index["chapters"], index["recording"]["duration"]),
            m["recording_id"],
            m["start"],
            m["end"],
        )
    except Exception as exc:
        raise friendly(exc) from exc


def evaluation_template():
    data = []
    for recording in ENGINE.store.list():
        data.append(
            {
                "recording_id": recording["id"],
                "query": "REPLACE with your actual question",
                "present": True,
                "expected_start": 0,
                "expected_end": min(10, recording["duration"]),
            }
        )
    path = ENGINE.store.root / "evaluation_template.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


def backup_library():
    from .engine import LOCK

    try:
        with LOCK:
            return ENGINE.store.backup()
    except Exception as exc:
        raise friendly(exc) from exc


def restore_library(file):
    from .engine import LOCK

    try:
        if not file:
            raise ValueError("Upload a library backup ZIP first.")
        with LOCK:
            ENGINE.store.restore(file)
        return *refresh(), "Library restored. Saved recordings and indexes are available again."
    except Exception as exc:
        raise friendly(exc) from exc


def labels_view(rid):
    require_recording(rid)
    return [
        [
            c["query"],
            c["present"],
            c.get("expected_start"),
            c.get("expected_end"),
            c["source"],
            c["reviewed"],
        ]
        for c in ENGINE.store.cases([rid])
    ]


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value in (1, "true", "True", "yes", "Yes"):
        return True
    if value in (0, None, "", "false", "False", "no", "No"):
        return False
    raise ValueError("Present and Reviewed must be true or false.")


def save_labels(rid, rows):
    try:
        require_recording(rid)
        if hasattr(rows, "values"):
            rows = rows.values.tolist()
        cases = []
        for row in rows:
            if not row or not str(row[0] or "").strip():
                continue
            query, present, start, end, source, reviewed = row
            present = as_bool(present)
            case = dict(
                recording_id=rid,
                query=str(query).strip(),
                present=present,
                source=str(source or "manual"),
                reviewed=as_bool(reviewed),
            )
            if present:
                case.update(expected_start=float(start), expected_end=float(end))
            cases.append(case)
        ENGINE.store.save_cases(rid, cases)
        return (
            labels_view(rid),
            f"Saved {len(cases)} answer labels. Future evaluations reuse them; changed labels invalidate cached reports.",
        )
    except Exception as exc:
        raise friendly(exc) from exc


def import_labels(file, rid):
    try:
        if not file:
            raise ValueError("Choose a label JSON file to import once.")
        cases = json.loads(Path(file).read_text())
        validate_cases(cases, ENGINE.store)
        for recording_id in set(c["recording_id"] for c in cases):
            ENGINE.store.save_cases(
                recording_id, [c for c in cases if c["recording_id"] == recording_id]
            )
        return (
            labels_view(rid) if rid else [],
            f"Imported {len(cases)} labels into the saved library. You do not need to upload this file again.",
        )
    except Exception as exc:
        raise friendly(exc) from exc


def evaluate(scope, methods, force=False, progress=gr.Progress()):
    try:
        summaries, csv_path, json_path, cached = run_saved_evaluation(
            ENGINE, scope or [], methods, progress, bool(force)
        )
        rows = [
            [
                s["method"],
                s["positive_queries"],
                s["absent_queries"],
                s["hit1"],
                s["hit3"],
                s["mrr3"],
                s["mean_context_coverage"],
                s["median_search_seconds"],
                s["absent_queries_returning_candidates"],
            ]
            for s in summaries
        ]
        unreviewed = sum(not c.get("reviewed", False) for c in ENGINE.store.cases(scope or []))
        message = (
            "Loaded cached evaluation report."
            if cached
            else "Evaluated using saved answer labels and cached recording indexes."
        )
        if unreviewed:
            message += f" {unreviewed} labels have not been marked reviewed; spot-check caption alignment before relying on metrics."
        return rows, csv_path, json_path, message
    except Exception as exc:
        raise friendly(exc) from exc


def automatic_evaluation(enabled, scope, methods, progress=gr.Progress()):
    if not enabled:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    if not scope or not ENGINE.store.cases(scope):
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            "Index updated. Automatic evaluation skipped because no saved labels exist for the selected recordings.",
        )
    return evaluate(scope, methods, False, progress)


def build_demo():
    with gr.Blocks(title="Find the Moment", css=CSS, js=JS) as demo:
        hits = gr.State([])
        bookmark_recording = gr.State(None)
        gr.Markdown(
            "# Find the Moment\nA saved media library for finding explanations, examples, and sounds."
        )
        status = gr.Markdown()
        with gr.Tab("Library and chapters"):
            with gr.Row():
                upload = gr.File(
                    label="Upload audio/video",
                    type="filepath",
                    file_types=[
                        ".wav",
                        ".mp3",
                        ".m4a",
                        ".flac",
                        ".ogg",
                        ".aac",
                        ".mp4",
                        ".mov",
                        ".mkv",
                        ".webm",
                    ],
                )
                url = gr.Textbox(label="Or public YouTube/direct media URL")
            generated = gr.Checkbox(
                value=True,
                label="Generate readable chapter titles and summaries locally (extra model download/time)",
            )
            import_button = gr.Button("Import and process", variant="primary")
            gr.Markdown(
                "Use one source at a time. YouTube downloads are best effort; upload if blocked. "
                "Transcription uses CPU int8; embeddings use a GPU when available. "
                "This is a single-owner library: all authorized users see saved recordings and bookmarks."
            )
            with gr.Row():
                active = gr.Dropdown(choices=choices(), label="Saved recording")
                refresh_button = gr.Button("Refresh saved library")
                process_button = gr.Button("Process selected recording again")
            stats = gr.Markdown()
            chapter_table = gr.Dataframe(
                headers=[
                    "Section",
                    "Start seconds",
                    "End seconds",
                    "Title",
                    "Likely type",
                    "Summary",
                    "Metadata source",
                ],
                datatype=["number", "number", "number", "str", "str", "str", "str"],
                interactive=False,
            )
            chapter_choice = gr.Dropdown(choices=[], label="Chapter to play or edit")
            play_chapter = gr.Button("Play selected chapter")
            with gr.Accordion("Edit chapter titles and boundaries", open=False):
                title = gr.Textbox(label="Title")
                label = gr.Dropdown(LABELS, value="Discussion", label="Section type")
                with gr.Row():
                    start = gr.Number(label="Start seconds")
                    end = gr.Number(label="End seconds")
                save_chapter = gr.Button("Save chapter changes")
                split_time = gr.Number(label="Split at seconds (snaps to transcript boundary)")
                with gr.Row():
                    split = gr.Button("Split chapter")
                    merge = gr.Button("Merge with following chapter")
            with gr.Accordion("Correct transcript and reindex", open=False):
                gr.Markdown(
                    "Edit text and timestamps below. Saving rebuilds chapters and resets current chapter edits; previous index versions remain on disk."
                )
                transcript_editor = gr.Dataframe(
                    headers=["Start seconds", "End seconds", "Text"],
                    datatype=["number", "number", "str"],
                    type="array",
                    interactive=True,
                )
                save_transcript = gr.Button("Save corrections and rebuild index")
            transcript_file = gr.File(label="Timestamped transcript", interactive=False)
            chapters_file = gr.File(label="Chapters JSON", interactive=False)
            with gr.Accordion("Back up or restore the recording library", open=False):
                gr.Markdown(
                    "A backup includes audio, transcripts, indexes, chapters, and bookmarks. Restore works only into an empty library. "
                    "Colab disks and ordinary Space disks are temporary; keep a backup or configure durable storage."
                )
                backup_button = gr.Button("Create portable library backup")
                backup_file = gr.File(label="Download library backup", interactive=False)
                restore_file = gr.File(
                    label="Upload a library backup ZIP", type="filepath", file_types=[".zip"]
                )
                restore_button = gr.Button("Restore into empty library")
        with gr.Tab("Search"):
            scope = gr.Dropdown(choices=choices(), multiselect=True, label="Recordings to search")
            query = gr.Textbox(
                label="What do you want to find?",
                placeholder="Find the worked example explaining overfitting",
            )
            with gr.Row():
                mode = gr.Radio(
                    ["Spoken content", "Sounds", "Both"],
                    value="Spoken content",
                    label="Search mode",
                )
                method = gr.Dropdown(
                    ["Keyword", "Semantic", "Hybrid", "Hybrid + reranker"],
                    value="Hybrid + reranker",
                    label="Speech ranking",
                )
            with gr.Row():
                seconds = gr.Slider(2, 30, 6, step=2, label="Sound window seconds")
                count = gr.Slider(1, 10, 5, step=1, label="Results")
                padding = gr.Slider(0, 15, 3, step=1, label="Extra playback context seconds")
            strategy = gr.Radio(
                ["Matched passage", "Explanation", "Topic section"],
                value="Explanation",
                label="Speech playback boundary strategy",
            )
            search_button = gr.Button("Search selected recordings", variant="primary")
            gr.Markdown(
                "Keyword search uses BM25. Hybrid combines BM25 and embeddings; the cross-encoder reranks the top 20 speech candidates. "
                "Both mode interleaves speech and sound rankings. Nearest matches are not proof the requested content exists."
            )
            results = gr.Dataframe(
                headers=[
                    "Rank",
                    "Recording",
                    "Mode",
                    "Likely type",
                    "Match start",
                    "Match end",
                    "Play start",
                    "Play end",
                    "Cosine similarity",
                    "Reranker score",
                    "Matched text",
                ],
                interactive=False,
            )
            result_choice = gr.Dropdown(choices=[], label="Result to play")
            open_button = gr.Button("Open selected result")
        with gr.Tab("Playback"):
            card = gr.HTML()
            offset = gr.HTML()
            clip = gr.Audio(label="Selected clip", elem_id="clip-player")
            timeline = gr.HTML()
            full_audio = gr.Audio(
                label="Full recording — click timeline or transcript to seek",
                elem_id="recording-player",
                type="filepath",
            )
            synced = gr.HTML()
            gr.Markdown(
                "The transcript highlights during full-recording or clip playback. Orange transcript borders indicate the matched passage; playback may include additional context."
            )
            with gr.Row():
                bookmark_start = gr.Number(label="Bookmark start seconds")
                bookmark_end = gr.Number(label="Bookmark end seconds")
            note = gr.Textbox(label="Bookmark note")
            save_mark = gr.Button("Save bookmark for current playback recording")
        with gr.Tab("Bookmarks"):
            marks = gr.Dataframe(headers=["Recording", "Start", "End", "Note"], interactive=False)
            mark_choice = gr.Dropdown(choices=[], label="Saved bookmark")
            with gr.Row():
                refresh_marks = gr.Button("Refresh bookmarks")
                play_mark = gr.Button("Play bookmark")
                delete_mark = gr.Button("Delete selected bookmark")
            marks_file = gr.File(label="Export bookmarks JSON", interactive=False)
        with gr.Tab("Evaluation"):
            gr.Markdown(
                "Answer labels are saved in your recording library; JSON is optional. "
                "For the matching full MIT Lecture 01 filename/duration, official caption-based labels are preloaded. "
                "Select a recording in Library and chapters, then load or edit its labels below. "
                "The app cannot know correct answers from its own predictions. "
                "Evaluation compares warm-query retrieval; it does not measure first-load or transcription time. "
                "Positive hits require overlap covering 25% of the shorter target/match interval. "
                "Absent-content queries report how many candidates were returned; no calibrated absence detector is claimed."
            )
            load_labels = gr.Button("Load labels for selected library recording")
            label_editor = gr.Dataframe(
                headers=[
                    "Query",
                    "Present",
                    "Expected start seconds",
                    "Expected end seconds",
                    "Label source",
                    "Reviewed",
                ],
                datatype=["str", "bool", "number", "number", "str", "bool"],
                type="array",
                interactive=True,
            )
            save_label_button = gr.Button("Save answer labels")
            with gr.Accordion("Optional JSON import/export", open=False):
                template_button = gr.Button("Download evaluation template for saved recordings")
                template_file = gr.File(interactive=False)
                eval_file = gr.File(
                    label="Import label JSON once", file_types=[".json"], type="filepath"
                )
                import_label_button = gr.Button("Save labels from JSON")
            eval_methods = gr.CheckboxGroup(
                ["Keyword", "Semantic", "Hybrid", "Hybrid + reranker"],
                value=["Keyword", "Hybrid + reranker"],
                label="Methods to compare",
            )
            force_eval = gr.Checkbox(
                value=False, label="Recompute even if a matching report is cached"
            )
            auto_eval = gr.Checkbox(
                value=False,
                label="Automatically evaluate after transcript or chapter edits (using Search-selected recordings)",
            )
            eval_button = gr.Button("Run saved evaluation over search-selected recordings")
            eval_table = gr.Dataframe(
                headers=[
                    "Method",
                    "Positive queries",
                    "Absent queries",
                    "Hit@1",
                    "Hit@3",
                    "MRR@3",
                    "Mean context coverage",
                    "Median search seconds",
                    "Absent queries returning candidates",
                ],
                interactive=False,
            )
            eval_csv = gr.File(label="Per-query results CSV", interactive=False)
            eval_json = gr.File(label="Summary JSON", interactive=False)
        browse = [
            chapter_table,
            chapter_choice,
            transcript_editor,
            transcript_file,
            chapters_file,
            full_audio,
            synced,
            timeline,
            stats,
        ]
        play_outputs = [
            clip,
            card,
            offset,
            full_audio,
            synced,
            timeline,
            bookmark_recording,
            bookmark_start,
            bookmark_end,
        ]
        shared = {"concurrency_limit": 1, "concurrency_id": "media-work"}
        refresh_button.click(refresh, [], [active, scope, *browse], **shared)
        demo.load(refresh, [], [active, scope, *browse], **shared)
        import_button.click(
            import_media, [upload, url, generated], [active, scope, *browse, status], **shared
        )
        active.input(view, [active], browse, **shared)
        process_button.click(process_selected, [active, generated], [*browse, status], **shared)
        chapter_choice.input(
            chapter_fields, [active, chapter_choice], [title, label, start, end], **shared
        )
        edit_events = [
            save_chapter.click(
                chapter_edit,
                [active, chapter_choice, title, label, start, end],
                [*browse, status],
                **shared,
            ),
            split.click(
                chapter_split, [active, chapter_choice, split_time], [*browse, status], **shared
            ),
            merge.click(chapter_merge, [active, chapter_choice], [*browse, status], **shared),
            save_transcript.click(
                correct, [active, transcript_editor, generated], [*browse, status], **shared
            ),
        ]
        play_chapter.click(play_section, [active, chapter_choice], play_outputs, **shared)
        search_button.click(
            search,
            [query, scope, mode, method, seconds, count, strategy, padding],
            [hits, results, result_choice, status],
            **shared,
        )
        open_button.click(open_result, [result_choice, hits], play_outputs, **shared)
        save_mark.click(
            save_bookmark,
            [bookmark_recording, bookmark_start, bookmark_end, note],
            [marks, mark_choice, marks_file, status],
            **shared,
        )
        refresh_marks.click(bookmarks_view, [], [marks, mark_choice, marks_file], **shared)
        delete_mark.click(
            delete_bookmark, [mark_choice], [marks, mark_choice, marks_file, status], **shared
        )
        play_mark.click(play_bookmark, [mark_choice], play_outputs, **shared)
        template_button.click(evaluation_template, [], [template_file], **shared)
        load_labels.click(labels_view, [active], [label_editor], **shared)
        save_label_button.click(
            save_labels, [active, label_editor], [label_editor, status], **shared
        )
        import_label_button.click(
            import_labels, [eval_file, active], [label_editor, status], **shared
        )
        eval_button.click(
            evaluate,
            [scope, eval_methods, force_eval],
            [eval_table, eval_csv, eval_json, status],
            **shared,
        )
        for event in edit_events:
            event.success(
                automatic_evaluation,
                [auto_eval, scope, eval_methods],
                [eval_table, eval_csv, eval_json, status],
                **shared,
            )
        backup_button.click(backup_library, [], [backup_file], **shared)
        restore_button.click(
            restore_library, [restore_file], [active, scope, *browse, status], **shared
        )
    return demo


def main():
    import secrets

    password = os.getenv("APP_PASSWORD")
    if os.getenv("COLAB_SHARE") == "1" and not password:
        password = secrets.token_urlsafe(12)
        print("Colab demo login username:", os.getenv("APP_USERNAME", "calvin"))
        print("Colab demo login password:", password, flush=True)
    auth = (os.getenv("APP_USERNAME", "calvin"), password) if password else None
    build_demo().queue().launch(auth=auth, share=os.getenv("COLAB_SHARE") == "1")
