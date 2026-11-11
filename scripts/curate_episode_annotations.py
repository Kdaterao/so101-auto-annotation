#!/usr/bin/env python3
"""Review segmented episodes in a local browser app and publish human curation.

This is a separate editor; scripts/view_episode_annotations.py remains read-only.
The app reads captioning/episodes sidecars and source videos, saves local drafts,
and optionally publishes curation/episodes sidecars to the same dataset repo.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import webbrowser

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from robot101.data.utilities.curation import (
    CURATION_DIR, SEGMENT_DIR, atomic_json, episode_filename, fingerprint,
    fresh_curation, remote_episode_json, validate_curation,
)
from robot101.data.utilities.episode_helpers import _load_lerobot, _parse_episodes


PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 manual episode curation</title>
<style>
:root{color-scheme:dark;font:15px system-ui;background:#0b1220;color:#e7eef8}*{box-sizing:border-box}body{margin:0}
header{height:62px;padding:12px 18px;background:#121d30;border-bottom:1px solid #2b3a52;display:flex;align-items:center;gap:14px}
h1{font-size:18px;margin:0}button,select,input,textarea{font:inherit;color:inherit;background:#1a2940;border:1px solid #40536e;border-radius:6px;padding:8px}
button{cursor:pointer}button:hover{background:#263b59}button.primary{background:#087b68;border-color:#13a58c}button.warn{background:#754b13;border-color:#aa7621}
#layout{height:calc(100vh - 62px);display:grid;grid-template-columns:290px minmax(450px,1fr) 340px}
aside{overflow:auto;padding:14px;background:#111a2a;border-right:1px solid #29384d}.episode{display:block;width:100%;text-align:left;margin:5px 0;padding:9px}.episode.active{border-color:#39c7a5;background:#173b3d}
.muted{color:#9fb0c7;font-size:13px}#main{overflow:auto;padding:16px}#right{overflow:auto;padding:16px;background:#111a2a;border-left:1px solid #29384d}
#videoWrap{position:relative;width:max-content;max-width:100%;margin:auto;background:#05080d;line-height:0}#video{display:block;max-width:min(100%,1100px);max-height:64vh;width:auto;height:auto}
#overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.panel{background:#131f32;border:1px solid #293b55;border-radius:9px;padding:14px;margin:12px 0}
#timeline{display:flex;gap:2px;min-height:42px}.segment{min-width:22px;overflow:hidden;padding:5px 2px;border-radius:4px;background:#2b5067}.segment.active{background:#087b68}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.row>*{margin:3px 0}textarea{width:100%;min-height:130px;resize:vertical;line-height:1.45}
.point{padding:8px;margin:6px 0;background:#1a2940;border-radius:6px}.point button{float:right;padding:4px 7px}.wide{width:100%}#status{min-height:22px;color:#f5ca65;white-space:pre-wrap}
label{font-size:13px;color:#b8c7dc}details{margin:10px 0}#original{white-space:pre-wrap;line-height:1.45}
@media(max-width:1050px){#layout{grid-template-columns:220px minmax(300px,1fr)}#right{grid-column:1/3;border-left:0;border-top:1px solid #29384d}}
@media(max-width:700px){#layout{height:auto;display:block}aside,#right,#main{max-height:none}#video{max-height:52vh}}
</style>
<header><h1>Manual curation</h1><span id="sourceLabel" class="muted"></span><span style="flex:1"></span><button id="save">Save draft</button><button id="publish" class="primary">Publish episode to Hugging Face</button></header>
<div id="layout"><aside><div class="row"><label>Queue</label><select id="queueFilter"><option value="all">All</option><option>todo</option><option>accepted</option><option>uncertain</option><option>skipped</option></select></div><p class="muted" id="counts"></p><div id="episodes"></div></aside>
<main id="main"><p id="task" class="muted"></p><div class="panel"><div class="row"><label>Segments</label><span id="segmentSummary" class="muted"></span></div><div id="timeline"></div><div id="boundaries"></div></div>
<div class="panel"><div id="videoWrap"><video id="video" controls preload="metadata" playsinline></video><canvas id="overlay"></canvas></div><p id="frameLabel" class="muted" style="text-align:center"></p></div>
<div id="status"></div></main>
<aside id="right"><h2 style="font-size:17px;margin-top:0">Selected segment</h2><label>Segment</label><select id="segmentSelect" class="wide"></select>
<div class="row"><label>Review status</label><select id="segmentStatus"><option>todo</option><option>accepted</option><option>uncertain</option><option>skipped</option></select></div>
<label for="annotation">Human annotation</label><textarea id="annotation" placeholder="Describe the action or state change for this segment"></textarea>
<div class="row"><button id="record">● Dictate</button><button id="stopRecord" disabled>Stop and transcribe</button><span id="micStatus" class="muted"></span></div>
<details id="modelBox"><summary>Model caption (reference)</summary><p id="original"></p></details>
<div class="panel"><label>POV point label</label><input id="pointLabel" class="wide" placeholder="e.g. red block"><p class="muted">Pause the video on the object and click it to add a tracking seed.</p><div id="points"></div></div>
<div class="row"><label>Episode status</label><select id="episodeStatus"><option>todo</option><option>accepted</option><option>uncertain</option><option>skipped</option></select></div>
<p class="muted">Click seeds are recorded in source-frame pixels on the wrist POV camera. Accepted seeds are propagated later by the separate BootsTAPIR command.</p></aside></div>
<script>
const $=id=>document.getElementById(id);let catalog=[],episode=null,curation=null,record=null,segmentId=null,videoInfo=null,dirty=false,mediaRecorder=null,audioParts=[],autosaveTimer=null;
const frameAt=()=>{if(!videoInfo)return 0;const t=$('video').currentTime-videoInfo.video_offset;const a=videoInfo.timestamps;let lo=0,hi=a.length-1;while(lo<hi){const mid=(lo+hi)>>1;if(a[mid]<t)lo=mid+1;else hi=mid}if(lo&&Math.abs(a[lo-1]-t)<Math.abs(a[lo]-t))lo--;return lo};
function selectedSegments(){return (record.selected_node_ids||[]).map(id=>record.nodes.find(n=>n.id===id)).filter(Boolean).sort((a,b)=>curation.segments[a.id].start_frame-curation.segments[b.id].start_frame)}
function activeAnn(){return curation?.segments?.[segmentId]}
function setDirty(){dirty=true;$('status').textContent='Unsaved edits · saving local draft…';clearTimeout(autosaveTimer);autosaveTimer=setTimeout(()=>save(false,true).catch(e=>$('status').textContent=`Draft save failed: ${e}`),900)}
function renderCatalog(){const filter=$('queueFilter').value;const counts={todo:0,accepted:0,uncertain:0,skipped:0};for(const e of catalog)counts[e.status||'todo']=(counts[e.status||'todo']||0)+1;$('counts').textContent=`${catalog.length} episodes · `+Object.entries(counts).map(([k,v])=>`${k}: ${v}`).join(' · ');$('episodes').replaceChildren();for(const e of catalog){if(filter!=='all'&&(e.status||'todo')!==filter)continue;const b=document.createElement('button');b.className='episode'+(e.id===episode?' active':'');b.textContent=`Episode ${e.id} · ${e.status||'todo'}`;b.onclick=()=>openEpisode(e.id);$('episodes').append(b)}}
function render(){if(!record||!curation)return;const segs=selectedSegments();if(!segs.some(n=>n.id===segmentId))segmentId=segs[0]?.id||null;$('task').textContent=`Episode ${episode} · ${curation.source_repo_id}@${curation.source_revision} · ${record.task||''}`;$('sourceLabel').textContent=`${catalog.length} segmented episodes · source pinned`;$('segmentSummary').textContent=`${segs.length} selected segments · [start, end) source frames`;
 $('timeline').replaceChildren();$('boundaries').replaceChildren();$('segmentSelect').replaceChildren();for(let i=0;i<segs.length;i++){const n=segs[i],a=curation.segments[n.id],button=document.createElement('button');button.className='segment'+(n.id===segmentId?' active':'');button.style.flex=String(a.end_frame-a.start_frame);button.textContent=n.id;button.title=`${n.id} · ${a.start_frame}–${a.end_frame-1}`;button.onclick=()=>selectSegment(n.id);$('timeline').append(button);$('segmentSelect').add(new Option(`${n.id} · [${a.start_frame}, ${a.end_frame})`,n.id));if(i<segs.length-1){const cut=document.createElement('label');cut.textContent=`${n.id} end / next start `;const input=document.createElement('input');input.type='number';input.min='1';input.max=String(record.n_frames-1);input.value=String(a.end_frame);input.style.width='100px';input.onchange=()=>changeBoundary(n.id,segs[i+1].id,Number(input.value));cut.append(input);$('boundaries').append(cut)}}
 const a=activeAnn(),n=record.nodes.find(x=>x.id===segmentId);if(!a||!n)return;$('segmentSelect').value=segmentId;$('segmentStatus').value=a.review_status;$('episodeStatus').value=curation.episode_review_status;$('annotation').value=a.text||'';$('original').textContent=n.caption?.text||'No model caption available.';$('modelBox').hidden=!n.caption?.text;$('points').replaceChildren();for(const point of a.points){const d=document.createElement('div');d.className='point';d.textContent=`${point.label} · frame ${point.seed_frame} · (${Number(point.xy[0]).toFixed(1)}, ${Number(point.xy[1]).toFixed(1)})`;const remove=document.createElement('button');remove.textContent='Remove';remove.onclick=()=>{a.points=a.points.filter(p=>p.id!==point.id);setDirty();renderPoints()};d.append(remove);$('points').append(d)}
 $('segmentSelect').value=segmentId;renderCatalog();drawOverlay();updateFrameLabel()}
function renderPoints(){const a=activeAnn();$('points').replaceChildren();for(const point of a.points){const d=document.createElement('div');d.className='point';d.textContent=`${point.label} · frame ${point.seed_frame} · (${Number(point.xy[0]).toFixed(1)}, ${Number(point.xy[1]).toFixed(1)})`;const remove=document.createElement('button');remove.textContent='Remove';remove.onclick=()=>{a.points=a.points.filter(p=>p.id!==point.id);setDirty();renderPoints();drawOverlay()};d.append(remove);$('points').append(d)}drawOverlay()}
function selectSegment(id){segmentId=id;const a=activeAnn();const ts=videoInfo?.timestamps?.[a.start_frame]??a.start_frame/record.fps;$('video').currentTime=(videoInfo?.video_offset||0)+ts;render()}
function changeBoundary(leftId,rightId,value){const left=curation.segments[leftId],right=curation.segments[rightId];if(!Number.isInteger(value)||value<=left.start_frame||value>=right.end_frame){$('status').textContent='Boundary must leave at least one frame in both adjacent segments.';render();return}left.end_frame=value;right.start_frame=value;for(const ann of [left,right])ann.points=ann.points.filter(p=>p.seed_frame>=ann.start_frame&&p.seed_frame<ann.end_frame);setDirty();render()}
async function openEpisode(id){try{if(dirty)await save(false,true);const r=await fetch(`/api/episode?id=${id}`);const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);episode=id;record=data.record;curation=data.curation;videoInfo=data.video;segmentId=record.selected_node_ids?.[0]||null;$('video').src=`/api/video?id=${id}&camera=wrist`;dirty=false;$('status').textContent='';render();}catch(e){$('status').textContent=String(e)}}
async function loadCatalog(){const r=await fetch('/api/catalog');const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);catalog=data.episodes;$('publish').disabled=!data.can_publish;$('publish').title=data.can_publish?'Upload this episode to the same Hugging Face dataset':'Restart with --push-to-hub to enable publishing';renderCatalog();if(catalog.length)await openEpisode(catalog[0].id)}
async function save(publish=false,automatic=false){if(!episode||!curation)return;curation.segments[segmentId].text=$('annotation').value;curation.segments[segmentId].review_status=$('segmentStatus').value;curation.episode_review_status=$('episodeStatus').value;const saveEpisode=episode;const r=await fetch(`/api/save?id=${episode}${publish?'&publish=1':''}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(curation)});const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);if(saveEpisode===episode)dirty=false;$('status').textContent=publish?'Saved locally and published to Hugging Face.':automatic?'Draft saved locally.':'Draft saved locally.';const item=catalog.find(e=>e.id===episode);if(item)item.status=curation.episode_review_status;renderCatalog()}
function drawOverlay(){const c=$('overlay'),v=$('video'),rect=v.getBoundingClientRect(),dpr=window.devicePixelRatio||1;c.width=Math.round(rect.width*dpr);c.height=Math.round(rect.height*dpr);c.style.width=`${rect.width}px`;c.style.height=`${rect.height}px`;const ctx=c.getContext('2d');ctx.scale(dpr,dpr);if(!v.videoWidth||!v.videoHeight)return;const a=activeAnn();if(!a)return;for(const p of a.points){const x=p.xy[0]*rect.width/v.videoWidth,y=p.xy[1]*rect.height/v.videoHeight;ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.fillStyle='#ffcb55';ctx.fill();ctx.strokeStyle='#111';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#fff';ctx.font='12px system-ui';ctx.fillText(p.label,x+9,y-8)}}
function updateFrameLabel(){if(!record||!videoInfo)return;const f=frameAt(),a=activeAnn();$('frameLabel').textContent=`Source frame ${f} · ${(videoInfo.timestamps[f]||0).toFixed(2)}s${a?` · selected interval [${a.start_frame}, ${a.end_frame})`:''}`}
$('video').addEventListener('timeupdate',updateFrameLabel);$('video').addEventListener('loadedmetadata',()=>{drawOverlay();const a=activeAnn();if(a)$('video').currentTime=(videoInfo?.video_offset||0)+(videoInfo?.timestamps?.[a.start_frame]??a.start_frame/record.fps)});window.addEventListener('resize',drawOverlay);
$('video').addEventListener('click',e=>{const label=$('pointLabel').value.trim();if(!label){$('status').textContent='Enter a point label before clicking the POV video.';return}const a=activeAnn(),f=frameAt();if(f<a.start_frame||f>=a.end_frame){$('status').textContent='Pause within the selected segment before adding a point.';return}const r=$('video').getBoundingClientRect(),x=(e.clientX-r.left)*$('video').videoWidth/r.width,y=(e.clientY-r.top)*$('video').videoHeight/r.height;if(x<0||y<0||x>=$('video').videoWidth||y>=$('video').videoHeight)return;a.points.push({id:crypto.randomUUID(),label,seed_frame:f,xy:[x,y],camera:'wrist'});setDirty();renderPoints()});
$('annotation').addEventListener('input',()=>{if(activeAnn())activeAnn().text=$('annotation').value;setDirty()});$('segmentStatus').onchange=()=>{activeAnn().review_status=$('segmentStatus').value;setDirty();renderCatalog()};$('episodeStatus').onchange=()=>{curation.episode_review_status=$('episodeStatus').value;setDirty();renderCatalog()};$('segmentSelect').onchange=()=>selectSegment($('segmentSelect').value);$('queueFilter').onchange=renderCatalog;
$('save').onclick=()=>save(false).catch(e=>$('status').textContent=String(e));$('publish').onclick=()=>save(true).catch(e=>$('status').textContent=String(e));
$('record').onclick=async()=>{try{const stream=await navigator.mediaDevices.getUserMedia({audio:true});audioParts=[];mediaRecorder=new MediaRecorder(stream);mediaRecorder.ondataavailable=e=>{if(e.data.size)audioParts.push(e.data)};mediaRecorder.onstop=async()=>{stream.getTracks().forEach(t=>t.stop());$('micStatus').textContent='Transcribing locally…';try{const blob=new Blob(audioParts,{type:mediaRecorder.mimeType||'audio/webm'});const r=await fetch('/api/transcribe',{method:'POST',headers:{'Content-Type':blob.type},body:blob});const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);const ta=$('annotation');ta.value=(ta.value?ta.value.trim()+' ':'')+data.text;activeAnn().text=ta.value;setDirty();$('micStatus').textContent='Transcript added; edit as needed.'}catch(e){$('micStatus').textContent=String(e)}};mediaRecorder.start();$('record').disabled=true;$('stopRecord').disabled=false;$('micStatus').textContent='Recording locally…'}catch(e){$('micStatus').textContent=String(e)}};
$('stopRecord').onclick=()=>{if(mediaRecorder?.state==='recording'){mediaRecorder.stop();$('record').disabled=false;$('stopRecord').disabled=true}};
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue=''}});loadCatalog().catch(e=>$('status').textContent=String(e));
</script></html>'''


class CurationApp:
    def __init__(self, args):
        from huggingface_hub import HfApi

        self.args = args
        self.api = HfApi()
        info = self.api.repo_info(args.repo_id, repo_type="dataset", revision=args.revision)
        self.revision = info.sha
        self.files = set(self.api.list_repo_files(args.repo_id, repo_type="dataset", revision=self.revision))
        self.cache_dir = (args.hub_cache_dir or args.output_dir / "hub_cache").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = args.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: dict[int, dict] = {}
        raw = [name for name in self.files if name.startswith(f"{SEGMENT_DIR}/ep") and name.endswith(".json")]
        for name in sorted(raw):
            match = re.fullmatch(rf"{re.escape(SEGMENT_DIR)}/ep(\d+)\.json", name)
            if not match:
                continue
            episode = int(match[1])
            if args.episodes is not None and episode not in args.episodes:
                continue
            record = remote_episode_json(args.repo_id, self.revision, name, self.cache_dir)
            self.records[episode] = record
        if args.episodes is not None:
            missing = args.episodes - self.records.keys()
            if missing:
                raise ValueError(f"Selected episode sidecars are missing: {sorted(missing)}")
        if not self.records:
            raise ValueError(f"No segmentation records found under {SEGMENT_DIR}/")
        self.curation_cache: dict[int, dict] = {}
        self.dataset_cache: dict[int, object] = {}
        self.dataset_lock = threading.RLock()
        self.transcriber = None
        self.LeRobotDataset, _, _, default_lerobot_root = _load_lerobot()
        self.lerobot_root = args.source_cache_dir or default_lerobot_root

    def curation_path(self, episode: int) -> Path:
        return self.output_dir / CURATION_DIR / episode_filename(episode)

    def load_curation(self, episode: int) -> dict:
        if episode in self.curation_cache:
            return self.curation_cache[episode]
        local = self.curation_path(episode)
        if local.is_file():
            curation = json.loads(local.read_text(encoding="utf-8"))
        else:
            remote_name = f"{CURATION_DIR}/{episode_filename(episode)}"
            if remote_name in self.files:
                curation = remote_episode_json(self.args.repo_id, self.revision, remote_name, self.cache_dir)
            else:
                curation = fresh_curation(self.records[episode], self.args.repo_id, self.revision)
        validate_curation(curation, self.records[episode])
        self.curation_cache[episode] = curation
        return curation

    def dataset(self, episode: int):
        with self.dataset_lock:
            if episode in self.dataset_cache:
                return self.dataset_cache[episode]
            record = self.records[episode]
            source_root = self.lerobot_root / record["source_repo_id"]
            kwargs = dict(episodes=[episode], root=source_root,
                          revision=record["source_revision"], video_backend=self.args.video_backend)
            try:
                data = self.LeRobotDataset(record["source_repo_id"], **kwargs)
            except ValueError as exc:
                if "corresponds to no data" not in str(exc):
                    raise
                data = self.LeRobotDataset(record["source_repo_id"], force_cache_sync=True, **kwargs)
            self.dataset_cache[episode] = data
            return data

    def video_info(self, episode: int, camera: str = "wrist") -> dict:
        from robot101.data.utilities.episode_helpers import _cam_key

        record, curation = self.records[episode], self.load_curation(episode)
        ds = self.dataset(episode)
        key = _cam_key(camera)
        if key not in ds.meta.features:
            raise ValueError(f"Source episode {episode} has no {camera!r} camera")
        relative = ds.meta.get_video_file_path(episode, key)
        path = (Path(ds.root) / relative).resolve()
        root = Path(ds.root).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"Source video is unavailable for episode {episode}, camera {camera}")
        offset = float(ds.meta.episodes[episode].get(f"videos/{key}/from_timestamp", 0.0))
        timestamps = record.get("timestamps")
        if not timestamps or len(timestamps) != curation["n_frames"]:
            timestamps = [i / float(record["fps"]) for i in range(curation["n_frames"])]
        if ds.num_frames != curation["n_frames"]:
            raise ValueError(f"Source frame count changed for episode {episode}")
        return dict(path=path, offset=offset, timestamps=timestamps, fps=float(record["fps"]))

    def save(self, episode: int, curation: dict, publish: bool = False) -> dict:
        record = self.records[episode]
        validate_curation(curation, record)
        if curation.get("segment_record_fingerprint") != fingerprint(record):
            raise ValueError("Curation is based on a different segmentation snapshot")
        path = self.curation_path(episode)
        atomic_json(path, curation)
        self.curation_cache[episode] = curation
        if publish:
            if not self.args.push_to_hub:
                raise ValueError("Start the app with --push-to-hub to publish curation")
            self.api.upload_file(path_or_fileobj=str(path), path_in_repo=f"{CURATION_DIR}/{episode_filename(episode)}",
                                 repo_id=self.args.repo_id, repo_type="dataset",
                                 commit_message=f"Curate SO-101 episode {episode}")
        return dict(saved=True, published=publish)

    def transcribe(self, audio: bytes, content_type: str) -> str:
        if not audio:
            raise ValueError("No audio was received")
        if len(audio) > 16 * 1024 * 1024:
            raise ValueError("Audio recording exceeds the 16 MiB limit")
        if self.transcriber is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("Install the optional local transcription dependency with `uv sync --extra curation`") from exc
            self.transcriber = WhisperModel(self.args.stt_model, device=self.args.stt_device,
                                            compute_type=self.args.stt_compute_type)
        suffix = ".webm" if "webm" in content_type else ".ogg" if "ogg" in content_type else ".wav"
        fd, path = tempfile.mkstemp(prefix="so101-dictation-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(audio)
            segments, _ = self.transcriber.transcribe(path, language="en", vad_filter=True)
            return " ".join(segment.text.strip() for segment in segments).strip()
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def handler_for(app: CurationApp):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, value: dict):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    body = PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/catalog":
                    episodes = []
                    for episode in sorted(app.records):
                        curation = app.load_curation(episode)
                        episodes.append(dict(id=episode, status=curation["episode_review_status"],
                                              segments=len(curation["segments"])))
                    self._json(200, dict(episodes=episodes, repo_id=app.args.repo_id,
                                         revision=app.revision, can_publish=app.args.push_to_hub))
                    return
                if parsed.path == "/api/episode":
                    episode = int(query["id"][0])
                    record = app.records[episode]
                    curation = app.load_curation(episode)
                    info = app.video_info(episode, "wrist")
                    self._json(200, dict(record=record, curation=curation,
                                         video=dict(video_offset=info["offset"], timestamps=info["timestamps"], fps=info["fps"])))
                    return
                if parsed.path == "/api/video":
                    episode = int(query["id"][0])
                    camera = query.get("camera", ["wrist"])[0]
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", camera):
                        raise ValueError("Invalid camera name")
                    path = app.video_info(episode, camera)["path"]
                    size = path.stat().st_size
                    range_header = self.headers.get("Range")
                    start, end = 0, size - 1
                    status = 200
                    if range_header:
                        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                        if not match:
                            self.send_error(416)
                            return
                        left, right = match.groups()
                        if left:
                            start = int(left)
                            end = min(int(right), size - 1) if right else size - 1
                        elif right:
                            start = max(0, size - int(right))
                        if start > end or start >= size:
                            self.send_error(416)
                            return
                        status = 206
                    length = end - start + 1
                    self.send_response(status)
                    self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(length))
                    if status == 206:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.end_headers()
                    with path.open("rb") as stream:
                        stream.seek(start)
                        remaining = length
                        while remaining:
                            chunk = stream.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                    return
                self.send_error(404)
            except (KeyError, ValueError, OSError, RuntimeError) as exc:
                self._json(400, dict(error=str(exc)))

        def do_POST(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if parsed.path == "/api/transcribe":
                    result = app.transcribe(self.rfile.read(length), self.headers.get("Content-Type", ""))
                    self._json(200, dict(text=result))
                    return
                if parsed.path == "/api/save":
                    episode = int(query["id"][0])
                    if episode not in app.records:
                        raise ValueError("Episode is not in this curation session")
                    if length > 8 * 1024 * 1024:
                        raise ValueError("Curation record exceeds the 8 MiB limit")
                    payload = json.loads(self.rfile.read(length))
                    published = query.get("publish", ["0"])[0] == "1"
                    self._json(200, app.save(episode, payload, published))
                    return
                self.send_error(404)
            except (KeyError, ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
                self._json(400, dict(error=str(exc)))

        def log_message(self, fmt, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="HF dataset containing captioning/episodes sidecars")
    parser.add_argument("--revision", default="main", help="Segment snapshot; pinned to its SHA on startup")
    parser.add_argument("--episodes", help="Optional episode IDs/ranges, e.g. 0-30:2,105")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manual_curation"))
    parser.add_argument("--hub-cache-dir", type=Path, help="Local cache for annotation sidecars")
    parser.add_argument("--source-cache-dir", type=Path, help="LeRobot source video cache root")
    parser.add_argument("--push-to-hub", action="store_true", help="Enable the UI's Publish button")
    parser.add_argument("--stt-model", default="base.en", help="Local faster-whisper model name/path")
    parser.add_argument("--stt-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--stt-compute-type", default="int8")
    parser.add_argument("--video-backend", choices=["pyav", "torchcodec"], default="pyav")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    try:
        args.episodes = set(_parse_episodes(args.episodes)) if args.episodes else None
        app = CurationApp(args)
    except Exception as exc:
        parser.error(str(exc))
    server = ThreadingHTTPServer((args.host, args.port), handler_for(app))
    address = f"http://{args.host}:{server.server_port}"
    print(f"Manual curation app: {address}\nDataset: {args.repo_id}@{app.revision}\n"
          f"Episodes: {len(app.records)} · local drafts: {app.output_dir}", flush=True)
    if args.open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
