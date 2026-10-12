# Hugging Face Spaces

Create a Gradio Space, then upload `app.py`, `find_the_moment/`, `pyproject.toml`, `requirements.txt`, `packages.txt`, and `README.md`. Keep the package folder intact. `requirements.txt` installs the package and its dependencies.

Prepend this metadata to the Space's README. Keep the GitHub README as ordinary Markdown.

```yaml
---
title: Find the Moment
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: "3.11"
pinned: false
---
```

Watch Build logs, then open the App tab. Check the plan and hardware requirements shown by Hugging Face before selecting compute. This repo does not provision or deploy hosting.

Start with a private Space. For a restricted public demo, set `APP_PASSWORD` as a Secret and optionally `APP_USERNAME` as a variable. `HF_TOKEN` is an optional Secret for model download authentication. Do not set `COLAB_SHARE` on Spaces.

The library uses SQLite and local files. Ordinary Space disks are temporary. Export a full-library backup or configure durable storage with suitable filesystem locking. Remote object-store mounts are better used for backups unless SQLite compatibility is established.

CPU hosting can be slow for long recordings, especially generated chapter metadata. Test short uploads before increasing `MAX_DURATION_SECONDS`. Moving to Spaces does not guarantee that YouTube will allow downloads.
