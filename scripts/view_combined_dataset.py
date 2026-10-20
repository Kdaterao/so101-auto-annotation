"""Streamlit inspector for robot101.data.combine_datasets outputs."""
from __future__ import annotations

import json
import argparse
import hashlib
import io
import base64
from fractions import Fraction
from pathlib import Path, PurePosixPath

import av
import pandas as pd
import streamlit as st

from view_grounded_subtasks import episode_metadata


def safe_file(root, repo, revision, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe artifact path: {name}")
    if root:
        local = Path(root).expanduser().resolve() / name
        if local.is_file():
            return local
        manifest_path = Path(root).expanduser().resolve() / "manifest.json"
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            return Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision))
        except EntryNotFoundError:
            manifest_path = Path(hf_hub_download(repo, "manifest.json", repo_type="dataset", revision=revision))
    from huggingface_hub import hf_hub_download
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config", {}).get("lightweight"):
            for source in manifest["inputs"]:
                prefix = source["path"].rstrip("/") + "/"
                if name.startswith(prefix):
                    return Path(hf_hub_download(source["repo_id"], name[len(prefix):],
                                               repo_type="dataset", revision=source["revision"]))
    raise FileNotFoundError(f"Artifact not available: {name}")


@st.cache_data(show_spinner=False, max_entries=3, ttl=300)
def catalog(root, repo, requested_revision):
    if root:
        root = str(Path(root).expanduser().resolve())
        if not (Path(root) / "subtasks.jsonl").is_file():
            if (Path(root) / "manifest.json").is_file():
                raise ValueError("Combining has not finished: only the input manifest exists. "
                                 "Resume the combine_datasets command and wait for its 'Saved … episodes' message, "
                                 "then refresh this viewer. Uploading is optional; omit --push-to-hub for local use.")
            raise ValueError("This folder is not a completed combined dataset; select a folder containing subtasks.jsonl.")
        revision = "local"
        names = [p.relative_to(Path(root)).as_posix() for p in Path(root).rglob("*") if p.is_file()]
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        revision = api.repo_info(repo, repo_type="dataset", revision=requested_revision).sha
        names = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    path = safe_file(root, repo, revision, "subtasks.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return revision, names, rows


@st.cache_data(show_spinner=False, max_entries=12)
def video_metadata(root, repo, revision, prefix, destination_episode, names):
    info = json.loads(safe_file(root, repo, revision, f"{prefix}/meta/info.json").read_text())
    parts = sorted(n for n in names if n.startswith(f"{prefix}/meta/episodes/") and n.endswith(".parquet"))
    table = episode_metadata(parts, {destination_episode},
                             lambda n: pd.read_parquet(safe_file(root, repo, revision, n)))
    row = table.iloc[0]
    cameras = {}
    for camera in ("top", "wrist", "side"):
        key = f"observation.images.{camera}"
        if info["features"].get(key, {}).get("dtype") != "video":
            continue
        relative = info["video_path"].format(video_key=key,
            chunk_index=int(row[f"videos/{key}/chunk_index"]),
            file_index=int(row[f"videos/{key}/file_index"]))
        cameras[camera] = dict(path=f"{prefix}/{relative}", offset=float(row[f"videos/{key}/from_timestamp"]))
    return float(info["fps"]), cameras


@st.cache_data(show_spinner=False, max_entries=6)
def playback_clip(path, file_stamp, offset, fps, start, end, camera, render_key, _renderer):
    """Decode once and render an exact half-open interval into a cached H.264 preview."""
    rate = Fraction(str(fps)).limit_denominator(100000)
    buffer = io.BytesIO()
    with av.open(path) as source, av.open(buffer, mode="w", format="mp4") as output:
        video = source.streams.video[0]
        source.seek(max(0, int((offset + start / fps) / video.time_base)), stream=video, backward=True)
        decoded = (frame for frame in source.decode(video) if frame.time is not None)
        previous, current = None, next(decoded, None)
        encoder = None
        for index, episode_frame in enumerate(range(start, end)):
            target = offset + episode_frame / fps
            while current is not None and float(current.time) < target:
                previous, current = current, next(decoded, None)
            choices = [frame for frame in (previous, current) if frame is not None]
            nearest = min(choices, key=lambda frame: abs(float(frame.time) - target)) if choices else None
            if nearest is None or abs(float(nearest.time) - target) > 1.5 / fps:
                raise ValueError(f"Missing aligned {camera} frame {episode_frame}; preview not fabricated")
            rgb = nearest.to_ndarray(format="rgb24")
            if _renderer is not None:
                rgb = _renderer.render(rgb, episode_frame, camera)
            if encoder is None:
                encoder = output.add_stream("libx264", rate=rate)
                encoder.width, encoder.height = rgb.shape[1], rgb.shape[0]
                encoder.pix_fmt = "yuv420p"
                encoder.options = {"crf": "20", "preset": "veryfast"}
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = index, 1 / rate
            for packet in encoder.encode(frame):
                output.mux(packet)
        if encoder is None:
            raise ValueError("Empty subtask preview")
        for packet in encoder.encode():
            output.mux(packet)
    return buffer.getvalue()


def episode_player(clips, records, fps, selected_start):
    """One browser component owns both cameras and their shared playback clock."""
    import streamlit.components.v1 as components
    timeline = [dict(start=r["start_frame"], end=r["end_frame"],
                     caption=r.get("detailed_caption") or r.get("brief_caption") or "No revised caption available") for r in records]
    sources = [dict(name=camera, src="data:video/mp4;base64," + base64.b64encode(clip).decode("ascii"))
               for camera, clip in clips]
    html = r'''<!doctype html><meta charset="utf-8">
<style>body{font:15px system-ui;margin:0;color:#ddd;background:#0e1117}
#grid{display:grid;grid-template-columns:repeat(CAMERA_COUNT,minmax(0,1fr));gap:16px}
.caption{height:150px;overflow:auto;padding:10px;background:#1c2230;border-radius:6px;margin-bottom:8px;white-space:pre-wrap}
video{width:100%;display:block;background:black}#controls{display:flex;gap:12px;align-items:center;margin-top:12px}
button{padding:8px 18px;background:#263248;color:white;border:1px solid #64748b;border-radius:5px;cursor:pointer}
input{flex:1;min-width:40px}#status{font-size:13px;margin-top:8px;color:#cbd5e1}
@media(max-width:650px){#grid{grid-template-columns:1fr}.caption{height:100px}}</style>
<div id="grid"></div><div id="controls"><button id="play" disabled>Loading…</button>
<input id="seek" type="range" min="0" value="0" max="1" step="1" disabled aria-label="Shared episode timeline">
<span id="time">0:00 / 0:00</span></div><div id="status">Loading both camera videos…</div>
<script>
const timeline=TIMELINE_DATA,sources=SOURCES_DATA,fps=FPS_DATA;
const grid=document.getElementById('grid'),button=document.getElementById('play'),slider=document.getElementById('seek'),status=document.getElementById('status');
let ready=false,wantsPlay=false,starting=false,seeking=false,scrubbing=false,serial=0,duration=0,lastCaption=-2;
const videos=[],captions=[];
function pauseAll(){videos.forEach(v=>{v.pause();v.playbackRate=1});}
function clock(t){const s=Math.max(0,Math.floor(t));return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;}
function captionAt(t){const f=Math.floor(t*fps+1e-5),i=timeline.findIndex(s=>f>=s.start&&f<s.end);if(i===lastCaption)return;lastCaption=i;captions.forEach(c=>c.textContent=timeline[i]?.caption||'No revised caption at this time');}
function display(t){captionAt(t);if(!scrubbing)slider.value=Math.round(t*fps);document.getElementById('time').textContent=`${clock(t)} / ${clock(duration)}`;button.textContent=wantsPlay?'Pause':'Play';}
function fail(message){wantsPlay=false;pauseAll();button.disabled=true;slider.disabled=true;status.textContent=message;}
function buffered(){return videos.every(v=>v.readyState>=3&&!v.seeking);}
async function playAll(){if(!ready||!wantsPlay||starting||seeking||scrubbing)return;if(!buffered()){pauseAll();status.textContent='Waiting for both cameras to buffer…';return;}
starting=true;try{await Promise.all(videos.map(v=>v.play()));if(!wantsPlay||seeking)pauseAll();else status.textContent='';}
catch(e){wantsPlay=false;pauseAll();status.textContent='Click Play to start both cameras.';}finally{starting=false;display(videos[0].currentTime);}}
function setPosition(video,t){if(Math.abs(video.currentTime-t)<.001&&!video.seeking)return Promise.resolve();return new Promise((resolve,reject)=>{
const done=()=>{video.removeEventListener('seeked',done);video.removeEventListener('error',bad);resolve();},bad=()=>{video.removeEventListener('seeked',done);reject(Error('Video seek failed'));};
video.addEventListener('seeked',done);video.addEventListener('error',bad,{once:true});video.currentTime=t;});}
async function seekAll(t){const request=++serial;seeking=true;pauseAll();t=Math.max(0,Math.min(t,Math.max(0,duration-1/fps)));captionAt(t);
try{await Promise.all(videos.map(v=>setPosition(v,t)));if(request!==serial)return;seeking=false;display(t);await playAll();}
catch(e){if(request===serial){seeking=false;fail(e.message);}}}
const loaders=sources.map(source=>new Promise((resolve,reject)=>{
const pane=document.createElement('div'),caption=document.createElement('div'),video=document.createElement('video');caption.className='caption';caption.textContent='Loading revised caption…';
video.muted=true;video.playsInline=true;video.preload='auto';video.setAttribute('aria-label',source.name+' camera');
pane.append(caption,video);grid.append(pane);captions.push(caption);videos.push(video);
let loaded=false;function check(){if(!loaded&&video.readyState>=4){loaded=true;resolve();}if(ready)playAll();}
video.addEventListener('canplaythrough',check);video.addEventListener('canplay',check);video.addEventListener('seeked',()=>{if(ready&&!seeking)playAll();});
video.addEventListener('waiting',()=>{if(ready&&wantsPlay){pauseAll();status.textContent='Buffering: both cameras paused…';}});
video.addEventListener('ended',()=>{wantsPlay=false;pauseAll();display(duration);});
video.addEventListener('error',()=>{const error=Error(source.name+': video failed to load');reject(error);fail(error.message);});
video.src=source.src;video.load();check();}));
Promise.all(loaders).then(async()=>{duration=Math.min(...videos.map(v=>v.duration));if(!Number.isFinite(duration)||duration<=0)throw Error('Invalid camera duration');
ready=true;button.disabled=false;slider.disabled=false;slider.max=Math.max(0,Math.floor(duration*fps)-1);wantsPlay=true;await seekAll(START_DATA/fps);}).catch(e=>fail(e.message));
button.onclick=()=>{if(!ready)return;wantsPlay=!wantsPlay;if(wantsPlay){if(videos[0].ended||videos[0].currentTime>=duration-1/fps)seekAll(0);else playAll();}else pauseAll();display(videos[0].currentTime);};
slider.oninput=()=>{scrubbing=true;pauseAll();captionAt(Number(slider.value)/fps);};
slider.onchange=()=>{scrubbing=false;seekAll(Number(slider.value)/fps);};
function tick(){if(ready&&!seeking&&!scrubbing){const master=videos[0],t=master.currentTime;
if(wantsPlay&&!master.paused){const drift=videos.slice(1).some(v=>Math.abs(v.currentTime-t)>Math.max(1/fps,.04));if(drift){seekAll(t);}else videos.slice(1).forEach(v=>{const delta=v.currentTime-t;v.playbackRate=delta>.008?.98:delta<-.008?1.02:1;});}
display(t);}requestAnimationFrame(tick);}tick();
</script>'''
    html = (html.replace("TIMELINE_DATA", json.dumps(timeline).replace("<", "\\u003c"))
            .replace("SOURCES_DATA", json.dumps(sources).replace("<", "\\u003c"))
            .replace("FPS_DATA", str(float(fps))).replace("START_DATA", str(int(selected_start)))
            .replace("CAMERA_COUNT", str(len(clips))))
    components.html(html, height=760, scrolling=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--root", help="Completed local combined dataset folder")
    inputs.add_argument("--repo-id", help="Combined Hugging Face dataset")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--transition-display-seconds", type=float, default=0.5,
                        help="Display previous-interval cyan dots for this many seconds")
    parser.add_argument("--show-masks", action="store_true", help="Show SAM masks (hidden by default)")
    args = parser.parse_args()
    if not 0 <= args.transition_display_seconds < float("inf"):
        parser.error("--transition-display-seconds must be finite and nonnegative")
    default_root = Path("outputs/so101_spread73_v1_unified")
    root = args.root or (str(default_root) if not args.repo_id and (default_root / "subtasks.jsonl").is_file() else "")
    repo = args.repo_id or "kdaterao/so101_spread73_v1_unified"
    st.set_page_config(page_title="Robot episodes", layout="wide", initial_sidebar_state="collapsed")
    try:
        resolved, names, rows = catalog(root, repo, args.revision)
        keys = sorted({(r["source_repo_id"], r["source_revision"], r["episode_index"]) for r in rows})
        if not keys:
            st.info("No source episodes available.")
            return
        multiple_sources = len({k[:2] for k in keys}) > 1
        selected = st.selectbox("Source episode", keys, format_func=lambda k:
            f"Episode {k[2]}" + (f" · {k[0]} · {k[1][:10]}" if multiple_sources else ""))
        records = sorted([r for r in rows if (
            r["source_repo_id"], r["source_revision"], r["episode_index"]) == selected],
            key=lambda r: r["start_frame"])
        record = next((r for r in records if r.get("grounding_path") and r.get("grounding_dataset_path")), None)
        if record is None:
            st.info("No grounded video is available for this source episode.")
            return
        prefix = record["grounding_dataset_path"]
        fps, cameras = video_metadata(root, repo, resolved, prefix,
                                      int(record["grounding_destination_episode"]), tuple(names))
        report_path = safe_file(root, repo, resolved, record["grounding_path"])
        report = json.loads(report_path.read_text())
        if not report.get("subtasks"):
            st.info("No grounding intervals available for this episode.")
            return
        renderer = None
        render_hash = hashlib.sha256(json.dumps(report, sort_keys=True).encode())
        render_hash.update(f"{args.show_masks}:{args.transition_display_seconds}".encode())
        if not report.get("debug_overlays", False):
            from robot101.data.utilities.debug_overlays import DebugOverlayRenderer
            import tempfile
            import shutil
            with tempfile.TemporaryDirectory(prefix="robot101-view-masks-") as directory:
                copied = set()
                for entry in report["subtasks"]:
                    info = entry.get("pov", {}).get("gripper_filter", {})
                    if args.show_masks and info.get("status") == "ok" and info.get("mask_file"):
                        if info['mask_file'] in copied:
                            continue
                        copied.add(info['mask_file'])
                        source = safe_file(root, repo, resolved, f"{prefix}/point_tracks/{info['mask_file']}")
                        target = Path(directory) / info["mask_file"]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                        render_hash.update(source.read_bytes())
                renderer = DebugOverlayRenderer(report["subtasks"], Path(directory),
                    show_masks=args.show_masks, transition_seconds=args.transition_display_seconds, fps=fps)
        if report.get("debug_overlays", False):
            st.warning("This source video already contains overlays; baked-in masks and points cannot be removed by viewer settings.")
        chosen = [camera for camera in ("top", "wrist") if camera in cameras]
        if not chosen:
            chosen = list(cameras)[:2]
        if not chosen:
            st.info("No camera videos are available for this episode.")
            return
        clips = []
        for camera in chosen:
            try:
                video = cameras[camera]
                with st.spinner(f"Preparing {camera} playback…"):
                    path = safe_file(root, repo, resolved, video["path"])
                    clip = playback_clip(str(path), (path.stat().st_size, path.stat().st_mtime_ns),
                        video["offset"], fps, 0, max(r["end_frame"] for r in report["subtasks"]) + 1,
                        camera, render_hash.hexdigest(), renderer)
                clips.append((camera, clip))
            except Exception as exc:
                st.error(f"{camera}: {exc}")
                return
        episode_player(clips, records, fps, 0)
    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
