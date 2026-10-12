"""Safe HTML rendering and synchronized transcript/timeline behavior."""

import html

CSS = """
.transcript-scroll{max-height:420px;overflow:auto;line-height:1.6}
.transcript-row{display:block;width:100%;text-align:left;border:0;background:transparent;padding:7px;border-radius:6px;cursor:pointer;color:inherit}
.transcript-row.active{background:#dbeafe;color:#172554}
.transcript-row.match{border-left:4px solid #f59e0b}
.chapter-bar{height:24px;border-radius:5px;background:#2563eb;border:0;cursor:pointer;min-width:2px}
.timeline{display:flex;gap:2px;width:100%;margin:10px 0}
.timeline-legend{font-size:12px;color:#64748b}
.seek-match{background:#f59e0b;color:#111;padding:5px;border-radius:6px;margin:3px;border:0;cursor:pointer}
"""

JS = """() => {
  if(window.__momentSync) return;
  window.__momentSync=true;
  let lastActive=null;
  const bound=new WeakSet();
  function attach(){
    document.querySelectorAll('#recording-player audio, #clip-player audio').forEach(audio=>{
      if(bound.has(audio))return;bound.add(audio);
      audio.addEventListener('play',()=>{
        document.querySelectorAll('#recording-player audio, #clip-player audio').forEach(other=>{
          if(other!==audio)other.pause();
        });
      });
      audio.addEventListener('timeupdate',()=>{
        const offset=audio.closest('#clip-player') ? Number(document.querySelector('#playback-offset')?.dataset.offset||0) : 0;
        const seconds=audio.currentTime+offset;
        let active=null;
        document.querySelectorAll('.transcript-row').forEach(row=>{
          const yes=seconds>=Number(row.dataset.start)&&seconds<Number(row.dataset.end);
          row.classList.toggle('active',yes);if(yes)active=row;
        });
        if(active&&active!==lastActive){
          const box=active.closest('.transcript-scroll');
          if(box)box.scrollTop=Math.max(0,active.offsetTop-box.offsetTop-box.clientHeight/3);
          lastActive=active;
        }
      });
    });
  }
  document.addEventListener('click',event=>{
    const button=event.target.closest('[data-recording-seek]');
    if(!button)return;
    const audio=document.querySelector('#recording-player audio');
    if(!audio)return;
    const seek=Number(button.dataset.recordingSeek);
    const go=()=>{audio.currentTime=seek;audio.play().catch(()=>{});};
    if(audio.readyState)go();else audio.addEventListener('loadedmetadata',go,{once:true});
  });
  new MutationObserver(attach).observe(document.body,{childList:true,subtree:true});
  attach();
}"""


def transcript_html(segments, hit=None):
    rows = []
    for segment in segments:
        match = hit and segment["start"] < hit["end"] and segment["end"] > hit["start"]
        rows.append(
            f'<button type="button" class="transcript-row{" match" if match else ""}" '
            f'data-start="{segment["start"]:.3f}" data-end="{segment["end"]:.3f}" '
            f'data-recording-seek="{segment["start"]:.3f}">'
            f"<strong>{segment['start'] / 60:.2f} min</strong> "
            f"{html.escape(segment['text'])}</button>"
        )
    return '<div class="transcript-scroll">' + "".join(rows) + "</div>"


def timeline_html(chapters, duration, hits=None):
    bars = []
    position = 0
    for chapter in chapters:
        if chapter["start"] > position:
            bars.append(f'<div style="flex:{chapter["start"] - position}"></div>')
        title = html.escape(
            f"{chapter['title']} ({chapter['start']:.1f}–{chapter['end']:.1f}s)", quote=True
        )
        bars.append(
            f'<button type="button" class="chapter-bar" style="flex:{chapter["end"] - chapter["start"]}" '
            f'title="{title}" data-recording-seek="{chapter["start"]}"></button>'
        )
        position = chapter["end"]
    if duration > position:
        bars.append(f'<div style="flex:{duration - position}"></div>')
    matches = "".join(
        f'<button type="button" class="seek-match" data-recording-seek="{p["start"]}">'
        f"Match {i + 1}: {p['start']:.1f}s</button>"
        for i, p in enumerate(hits or [])
    )
    return (
        '<div class="timeline-legend">Topic timeline · click a blue section to seek the full recording</div>'
        '<div class="timeline">' + "".join(bars) + "</div>" + matches
    )
