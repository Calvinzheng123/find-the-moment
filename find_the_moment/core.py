"""Dependency-light segmentation, ranking, playback ranges, and editing rules."""

import math
import re
from collections import Counter

import numpy as np

LABELS = [
    "Discussion",
    "Definition",
    "Worked example",
    "Derivation",
    "Common mistake",
    "Question and answer",
    "Assignment",
    "Conclusion",
]


def validate_segments(rows, duration):
    segments = []
    previous_end = 0
    for number, row in enumerate(rows, 1):
        if len(row) != 3:
            raise ValueError("Transcript rows need Start seconds, End seconds, and Text.")
        a, b = float(row[0]), float(row[1])
        text = str(row[2]).strip()
        if not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b <= duration + 0.05:
            raise ValueError(f"Invalid timestamps in transcript row {number}.")
        if a < previous_end - 0.05:
            raise ValueError(f"Transcript row {number} overlaps or is out of order.")
        if not text:
            raise ValueError(f"Transcript row {number} is empty; remove it instead.")
        segments.append({"start": a, "end": min(b, duration), "text": text})
        previous_end = b
    return segments


def group_segments(segments, seconds=40, overlap=True):
    passages = []
    i = 0
    while i < len(segments):
        j = i + 1
        while j < len(segments) and segments[j]["end"] - segments[i]["start"] <= seconds:
            j += 1
        passages.append(
            {
                "start": segments[i]["start"],
                "end": segments[j - 1]["end"],
                "text": " ".join(s["text"] for s in segments[i:j]),
            }
        )
        if j == len(segments):
            break
        i = max(i + 1, j - 1) if overlap else j
    return passages


def fallback_title(text):
    # Use a source sentence rather than disconnected frequent keywords.
    sentence = next(
        (s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 12), text.strip()
    )
    words = sentence.split()
    return " ".join(words[:12]) + ("…" if len(words) > 12 else "")


def split_topics(units, vectors, min_seconds=60, max_seconds=210):
    if not units:
        return []
    similarities = [float(vectors[i - 1] @ vectors[i]) for i in range(1, len(units))]
    cutoff = float(np.quantile(similarities, 0.25)) if similarities else -1
    groups = []
    first = 0
    for i in range(1, len(units)):
        elapsed = units[i - 1]["end"] - units[first]["start"]
        force = units[i]["end"] - units[first]["start"] > max_seconds
        pause = units[i]["start"] - units[i - 1]["end"] >= 15
        if force or (elapsed >= min_seconds and (pause or similarities[i - 1] < cutoff)):
            groups.append((first, i))
            first = i
    groups.append((first, len(units)))
    if len(groups) > 1:
        a, b = groups[-1]
        p, _ = groups[-2]
        if (
            units[b - 1]["end"] - units[a]["start"] < 30
            and units[b - 1]["end"] - units[p]["start"] <= max_seconds
        ):
            groups[-2:] = [(p, b)]
    return [
        {
            "id": n,
            "start": units[a]["start"],
            "end": units[b - 1]["end"],
            "title": fallback_title(" ".join(u["text"] for u in units[a:b])),
            "summary": units[a]["text"][:400],
            "section_type": "Discussion",
            "metadata_source": "extractive",
            "reviewed": False,
        }
        for n, (a, b) in enumerate(groups)
    ]


def chapter_for(start, end, chapters):
    if not chapters:
        return None
    return (
        max(chapters, key=lambda c: max(0, min(end, c["end"]) - max(start, c["start"])))
        if any(start < c["end"] and end > c["start"] for c in chapters)
        else None
    )


def intent(query):
    rules = [
        ("Worked example", r"\b(example|walkthrough|demonstration|worked)\b"),
        ("Derivation", r"\b(derive|derivation|proof|prove)\b"),
        ("Definition", r"\b(define|definition)\b"),
        ("Assignment", r"\b(assignment|homework|deadline|due)\b"),
        ("Common mistake", r"\b(mistake|pitfall|misconception)\b"),
    ]
    return next((label for label, pattern in rules if re.search(pattern, query.lower())), None)


STOP = set(
    "the a an and or of to in is it that this we you i they for on with as are be was have has can will so but from at by not what how then there just our your its do if about would could should also into all which when now find explain".split()
)


def tokens(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOP]


def bm25(query, texts):
    """BM25 over the selected recording collection, computed per search."""
    docs = [tokens(text) for text in texts]
    n = len(docs)
    if not n:
        return np.array([], dtype=float)
    average = sum(map(len, docs)) / n or 1
    terms = set(tokens(query))
    df = {term: sum(term in set(doc) for doc in docs) for term in terms}
    scores = []
    for doc in docs:
        counts = Counter(doc)
        score = 0
        for term in terms:
            f = counts[term]
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * len(doc) / average)) if f else 0
        scores.append(score)
    return np.asarray(scores)


def retrieve(query, items, vectors, query_vector, method="Hybrid", limit=20):
    if not items:
        return []
    semantic = np.asarray(vectors) @ query_vector
    if not np.isfinite(semantic).all():
        raise ValueError("Invalid similarity values.")
    lexical = bm25(query, [p["text"] for p in items])
    sr = list(np.argsort(semantic)[::-1])
    lr = list(np.argsort(lexical)[::-1])
    if method == "Keyword":
        order = [i for i in lr if lexical[i] > 0]
    elif method == "Semantic":
        order = sr
    else:
        fused = {int(i): 1 / (60 + r) for r, i in enumerate(sr, 1)}
        for r, i in enumerate(lr, 1):
            if lexical[i] > 0:
                fused[int(i)] += 1 / (60 + r)
        target = intent(query)
        if target:
            for i in fused:
                if items[i].get("section_type") == target:
                    fused[i] += 0.005
        order = sorted(fused, key=lambda i: fused[i], reverse=True)
    return [
        dict(items[int(i)], similarity=float(semantic[i]), keyword_score=float(lexical[i]))
        for i in order[:limit]
    ]


def diverse(items, count):
    selected = []
    for item in items:
        # Deduplicate by actual match interval, not expanded context interval.
        if any(
            item["recording_id"] == p["recording_id"]
            and item["mode"] == p["mode"]
            and item["start"] < p["end"]
            and item["end"] > p["start"]
            for p in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= count:
            break
    return selected


def context_range(
    hit, segments, chapters, duration, strategy="Explanation", padding=3, max_seconds=240
):
    start, end = hit["start"], hit["end"]
    chapter = chapter_for(start, end, chapters)
    if strategy == "Topic section" and chapter:
        start, end = min(start, chapter["start"]), max(end, chapter["end"])
    elif strategy == "Explanation" and segments:
        # Prefer sentence endpoints and introductory/example transitions around the hit.
        left = max(0, hit["start"] - 75)
        right = min(duration, hit["end"] + 90)
        nearby = [s for s in segments if s["end"] > left and s["start"] < right]
        earlier = [
            s
            for s in nearby
            if s["start"] <= hit["start"]
            and re.search(
                r"\b(let.s|for example|suppose|consider|we define|next|now we|to understand)\b",
                s["text"].lower(),
            )
        ]
        if earlier:
            start = earlier[-1]["start"]
        else:
            before = [s for s in nearby if s["start"] <= hit["start"]]
            if before:
                start = before[max(0, len(before) - 3)]["start"]
        after = [s for s in nearby if s["end"] >= hit["end"]]
        end = after[min(2, len(after) - 1)]["end"] if after else end
        conclusions = [
            s
            for s in after
            if re.search(
                r"\b(therefore|in summary|so we get|this gives|as a result)\b", s["text"].lower()
            )
        ]
        if conclusions:
            end = conclusions[0]["end"]
        if chapter:
            start = max(min(chapter["start"], hit["start"]), start)
            end = min(max(chapter["end"], hit["end"]), end)
    if end - start > max_seconds:
        start = max(start, hit["start"] - max_seconds / 3)
        end = max(hit["end"], min(end, start + max_seconds))
    return max(0, start - padding), min(duration, end + padding)


def validate_chapters(chapters, duration):
    previous = 0
    for i, c in enumerate(chapters):
        a, b = float(c["start"]), float(c["end"])
        if not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b <= duration + 0.05:
            raise ValueError("Invalid chapter boundaries.")
        if a < previous - 0.05:
            raise ValueError("Chapters must remain ordered and nonoverlapping.")
        if not str(c["title"]).strip():
            raise ValueError("Chapter title cannot be empty.")
        if c["section_type"] not in LABELS:
            raise ValueError("Unknown section type.")
        c["id"] = i
        previous = b
    return chapters


def edit_chapter(chapters, index, title, label, start, end, duration):
    result = [dict(c) for c in chapters]
    c = result[int(index)]
    c.update(
        title=str(title).strip(),
        section_type=label,
        start=float(start),
        end=float(end),
        reviewed=True,
    )
    return validate_chapters(result, duration)


def split_chapter(chapters, index, point, segments, duration):
    result = [dict(c) for c in chapters]
    i = int(index)
    old = result[i]
    point = float(point)
    # Snap to an existing transcript boundary rather than splitting a sentence arbitrarily.
    boundaries = [s["start"] for s in segments if old["start"] < s["start"] < old["end"]]
    if not boundaries:
        raise ValueError("No internal transcript boundary available to split here.")
    if not old["start"] < point < old["end"]:
        raise ValueError("Split time must be inside the section.")
    point = min(boundaries, key=lambda t: abs(t - point))
    left = dict(old, end=point, title=old["title"] + " — part 1", reviewed=True)
    right = dict(old, start=point, title=old["title"] + " — part 2", reviewed=True)
    result[i : i + 1] = [left, right]
    return validate_chapters(result, duration)


def merge_chapters(chapters, index, duration):
    result = [dict(c) for c in chapters]
    i = int(index)
    if i + 1 >= len(result):
        raise ValueError("Select a section with a following section to merge.")
    a, b = result[i : i + 2]
    result[i : i + 2] = [
        dict(
            a,
            end=b["end"],
            title=a["title"] + " / " + b["title"],
            summary=a.get("summary", "") + " " + b.get("summary", ""),
            reviewed=True,
        )
    ]
    return validate_chapters(result, duration)
