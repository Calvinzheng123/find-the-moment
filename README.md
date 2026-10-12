# Find the Moment

Search lectures, podcasts, and video recordings for an explanation, example, or sound. Results include the matched transcript, timestamps, and a playable clip with optional surrounding context.

The app keeps a recording library so transcripts and indexes can be reused. It combines Whisper transcription, EmbeddingGemma 2 embeddings, BM25 keyword retrieval, and a cross-encoder reranker. Audio embeddings handle non-speech queries such as laughter or applause.

## Run locally

Requires Python 3.11–3.13 and FFmpeg. On macOS:

```bash
brew install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python app.py
```

Open the local URL printed in the terminal. Model weights download on first use. Start with a short recording and turn off generated chapter titles for a faster first test.

The current implementation uses CUDA for embeddings when available. Transcription, reranking, and title generation run on CPU. Apple GPU acceleration is not configured.

## Features

- Search one recording or a collection, using speech, sounds, or both.
- Browse automatic topic sections; rename, split, merge, or adjust them.
- Correct transcripts and rebuild their indexes.
- Play a short match, a wider explanation, or a full topic section.
- Navigate with a timeline and synchronized transcript.
- Save bookmarks and export or restore the recording library.
- Compare retrieval methods using saved answer labels and cached reports.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
```

Tests use controlled model outputs and do not download weights. They cover storage, retrieval, corrections, chapter edits, bookmarks, backups, and evaluation caching. Model quality must be measured on real recordings.

## Project layout

```text
app.py                 Local and Hugging Face entry point
find_the_moment/       Application package
tests/                Offline workflow tests
notebooks/             Colab runner
docs/                 Evaluation and deployment notes
```

Configuration is through environment variables; see [.env.example](.env.example). The app does not automatically read `.env` files.

## Notes

This is a single-owner library. Everyone with access to the app can see its saved recordings. Use a private Space or configure `APP_PASSWORD` when sharing it.

YouTube downloads can be blocked; file upload is the fallback. Video search uses audio, not slides or frames. Chapters and summaries need review, and nearest matches do not establish that requested content is present.

Saved data lasts only as long as the underlying disk. Back up the library before replacing a Colab runtime or hosting container.

See [evaluation](docs/evaluation.md), [Colab setup](docs/colab.md), and [Hugging Face deployment](docs/huggingface.md).
