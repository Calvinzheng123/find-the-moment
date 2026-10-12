# Colab

Open `notebooks/colab.ipynb` and select a T4 GPU in Runtime settings.

The notebook loads a GitHub repository URL or an uploaded repository ZIP, installs FFmpeg and the package, then launches the app in a separate Python process. Using a fresh process avoids reusing Gradio's event-loop state when relaunching.

Open the printed share link and use the printed login credentials. Keep the launch cell running. Stop it before changing files or launching again.

Use a short upload for the first test. The URL field must be empty when uploading a file. Generated chapter metadata is optional and takes additional time.

The notebook sets a two-hour recording limit. The saved library is in `/content/recording_library`, which disappears when Colab replaces the runtime. Export a library backup before that happens and restore it into an empty library later.

For the original full MIT Lecture 01 filename and duration, the app preloads caption-derived evaluation labels. Review them against playback before using the metrics.
