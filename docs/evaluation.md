# Evaluation

Use independent answer intervals to check retrieval. Generating labels from the system's own results would not test whether those results are correct.

## Run

1. Import and process a recording.
2. Select it in Library and chapters.
3. In Evaluation, load its saved labels or enter questions and answer intervals directly.
4. Check the intervals against the original audio and mark reviewed rows.
5. Select the recordings to evaluate in Search.
6. Compare Keyword, Semantic, Hybrid, and Hybrid + reranker.
7. Download the per-query CSV and summary JSON.

Labels are saved in SQLite. An unchanged combination of labels, methods, recordings, and index generations reuses the previous report. Select Recompute for a new timing measurement. Optional automatic evaluation runs after successful transcript or chapter edits.

The MIT Lecture 01 preset is installed only for the matching original filename and full duration. Its labels come from the official captions and are initially unreviewed. The source is [MIT 6.7960, Fall 2024, Lecture 01](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/resources/mit6_7960f24_lec01_mp4/). Trimmed recordings may have different timestamp offsets.

## Metrics

| Metric | Definition |
| --- | --- |
| Hit@1 | Fraction of positive queries whose first result overlaps the target |
| Hit@3 | Fraction whose first three results contain an overlapping target |
| MRR@3 | Mean reciprocal rank of the first correct result, up to rank three |
| Context coverage | Fraction of the labeled answer included in the first playback clip |
| Median search seconds | Median warm-query retrieval time |

A retrieval hit requires overlap covering at least 25% of the shorter target or matched interval. This is a permissive temporal criterion; inspect whether the retrieved text actually answers the question. It does not require playback to cover the full answer.

Absent-topic queries report how often candidates are returned. The app does not have a calibrated absence detector, so this is not negative-query accuracy. Search timing excludes ingestion, transcription, and initial model downloads.

## Initial test

A first run on one MIT lecture used eight positive queries and two absent-topic checks:

| Method | Hit@1 | Hit@3 | MRR@3 | Context coverage | Median search time |
| --- | --- | --- | --- | --- | --- |
| Keyword | 50% | 75% | 0.604 | 35.5% | 0.005 s |
| Hybrid + reranker | 87.5% | 87.5% | 0.875 | 63.4% | 0.318 s |

These values were transcribed from an initial run's UI report. They are a small caption-derived baseline, not a validated benchmark across recordings. The raw result files and hardware details have not been archived in this repo. Add those and test more recordings before making broader claims.

[evaluation-example.json](evaluation-example.json) shows the optional JSON import format. Its values are placeholders, not ground truth.
