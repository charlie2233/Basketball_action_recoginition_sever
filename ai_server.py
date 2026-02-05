"""
Minimal FastAPI inference server for the basketball action model.

Endpoints (matches the mobile app expectation):
  POST /api/ai/analyze { "videoUrl": "<path-or-url>" } -> { "jobId": "<uuid>" }
  GET  /api/ai/result/{jobId} -> AiResult with events and diagnostics

Usage:
  uvicorn ai_server:app --host 0.0.0.0 --port 8787
    (optional) set CHECKPOINT env var to override default checkpoint path.
    (optional) set CHECKPOINT_URL to download a checkpoint at startup.
    (optional) set LABELS_PATH to override the labels mapping json.

Notes:
  - Default checkpoint: model_checkpoints/r2plus1d_clean/r2plus1d_multiclass_13_0.0001.pt
  - Designed for local/edge inference: single-event per clip, covering full clip duration.
  - Remote URLs are downloaded to a temp file before scoring.
"""

import asyncio
import json
import mimetypes
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from torchvision import models
from torchvision.io import read_video

# Torchvision weights enum is newer; fall back to pretrained=True if unavailable.
try:
  from torchvision.models import R2Plus1D_18_Weights
  WEIGHTS = R2Plus1D_18_Weights.KINETICS400_V1
except Exception:
  R2Plus1D_18_Weights = None
  WEIGHTS = None

FALLBACK_LABELS = ['dunk', 'three_pointer', 'made_shot', 'block', 'steal', 'assist', 'foul']
DEFAULT_CKPT = Path("model_checkpoints/r2plus1d_clean/r2plus1d_multiclass_13_0.0001.pt")
DEFAULT_LABELS_PATH = Path("dataset/labels_dict.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
MIME_EXTENSION_MAP = {
  "video/quicktime": ".mov",
  "video/x-matroska": ".mkv",
  "video/webm": ".webm",
}

EVENT_LABEL_MAP = {
  "shoot": "made_shot",
  "shot": "made_shot",
  "made_shot": "made_shot",
  "block": "block",
  "pass": "assist",
  "assist": "assist",
  "dunk": "dunk",
  "three_pointer": "three_pointer",
  "three point": "three_pointer",
  "three-point": "three_pointer",
  "3 point": "three_pointer",
  "3-point": "three_pointer",
  "steal": "steal",
  "foul": "foul",
}
IGNORE_LABELS = {
  "run",
  "dribble",
  "walk",
  "ball in hand",
  "defense",
  "defence",
  "pick",
  "no_action",
  "discard",
  "no action",
}


class AnalyzeRequest(BaseModel):
  videoUrl: str
  options: Optional[dict] = None


class AiDiagnostics(BaseModel):
  processingMs: float
  avgScore: float


class AiEvent(BaseModel):
  id: str
  type: str
  start: float
  end: float
  confidence: float
  score: float


class AiResult(BaseModel):
  jobId: str
  status: str
  progress: int
  message: Optional[str] = None
  events: list[AiEvent]
  diagnostics: Optional[AiDiagnostics] = None


def load_labels() -> list[str]:
  override = os.environ.get("LABELS_PATH")
  labels_path = Path(override) if override else DEFAULT_LABELS_PATH

  if labels_path.exists():
    try:
      with labels_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
      if isinstance(data, list):
        return [str(item) for item in data]
      if isinstance(data, dict):
        items = sorted(
          ((int(key), value) for key, value in data.items()),
          key=lambda item: item[0],
        )
        return [str(value) for _, value in items]
    except Exception as exc:
      print(f"[warn] failed to load labels from {labels_path}: {exc}")

  return FALLBACK_LABELS.copy()


def map_label_to_event(raw_label: str) -> Optional[str]:
  label = raw_label.strip().lower()
  if label in IGNORE_LABELS:
    return None
  if label in EVENT_LABEL_MAP:
    return EVENT_LABEL_MAP[label]

  if "dunk" in label:
    return "dunk"
  if "three" in label and ("point" in label or "pointer" in label or "3" in label):
    return "three_pointer"
  if "shoot" in label or "shot" in label:
    return "made_shot"
  if "block" in label:
    return "block"
  if "steal" in label:
    return "steal"
  if "assist" in label or "pass" in label:
    return "assist"
  if "foul" in label:
    return "foul"

  return None


def download_file(url: str, destination: Path, chunk_size: int = 4 * 1024 * 1024) -> Path:
  timeout = int(os.getenv("AI_WORKER_DOWNLOAD_TIMEOUT", "60"))
  request = Request(url, headers={"User-Agent": "HoopsClips/1.0"})
  try:
    with urlopen(request, timeout=timeout) as response:
      with destination.open("wb") as handle:
        while True:
          chunk = response.read(chunk_size)
          if not chunk:
            break
          handle.write(chunk)
  except URLError as exc:
    raise RuntimeError(f"Failed to download {url}: {exc}") from exc
  return destination


def resolve_checkpoint_path() -> Path:
  override = os.environ.get("CHECKPOINT")
  if override:
    path = Path(override)
    if path.exists():
      return path
    raise FileNotFoundError(f"CHECKPOINT not found at {path}")

  if DEFAULT_CKPT.exists():
    return DEFAULT_CKPT

  remote = os.environ.get("CHECKPOINT_URL")
  if remote:
    tmp_dir = Path(tempfile.mkdtemp(prefix="ai-ckpt-"))
    dest = tmp_dir / "checkpoint.pt"
    return download_file(remote, dest)

  raise FileNotFoundError(
    "Checkpoint not found. Set CHECKPOINT to a local path or CHECKPOINT_URL to a remote file."
  )


def load_model(ckpt_path: Path, num_classes: int):
  if WEIGHTS is not None:
    model = models.video.r2plus1d_18(weights=WEIGHTS)
  else:
    model = models.video.r2plus1d_18(pretrained=True)
  state = torch.load(str(ckpt_path), map_location=DEVICE)
  sd = state.get("state_dict", state)
  # Infer num classes from checkpoint head shape if present
  ckpt_fc_weight = sd.get("fc.weight") if isinstance(sd, dict) else None
  if ckpt_fc_weight is not None:
    num_classes = ckpt_fc_weight.shape[0]
  num_ftrs = model.fc.in_features
  model.fc = torch.nn.Linear(num_ftrs, num_classes, bias=True)
  missing, unexpected = model.load_state_dict(sd, strict=False)
  if missing or unexpected:
    print(f"[warn] checkpoint load missing={missing}, unexpected={unexpected}")
  model.eval().to(DEVICE)
  return model, num_classes


def preprocess_frames(video: torch.Tensor) -> torch.Tensor:
  # video: (T, H, W, C) uint8
  if video.shape[0] == 0:
    raise ValueError("No frames in video")
  # sample up to 16 frames uniformly
  t = video.shape[0]
  idx = torch.linspace(0, t - 1, steps=min(16, t)).long()
  frames = video[idx]  # (t', H, W, C)
  frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (t', C, H, W)
  frames = torch.nn.functional.interpolate(frames, size=(112, 112))  # (t', C, 112, 112)
  frames = frames.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
  return frames.to(DEVICE)


def infer_extension(url: str, content_type: Optional[str] = None) -> str:
  parsed = urlparse(url)
  ext = Path(parsed.path or "").suffix.lower()
  if ext in VIDEO_EXTENSIONS:
    return ext

  if content_type:
    mime = content_type.split(";")[0].strip().lower()
    if mime in MIME_EXTENSION_MAP:
      return MIME_EXTENSION_MAP[mime]
    guessed = mimetypes.guess_extension(mime)
    if guessed and guessed in VIDEO_EXTENSIONS:
      return guessed

  return ".mp4"


def download_video(url: str, destination: Path, chunk_size: int = 524288) -> Path:
  timeout = int(os.getenv("AI_WORKER_DOWNLOAD_TIMEOUT", "60"))
  request = Request(url, headers={"User-Agent": "HoopsClips/1.0"})
  with urlopen(request, timeout=timeout) as response:
    content_type = response.headers.get("Content-Type")
    ext = infer_extension(url, content_type)
    dest = destination.with_suffix(ext) if destination.suffix.lower() != ext else destination
    with dest.open("wb") as handle:
      while True:
        chunk = response.read(chunk_size)
        if not chunk:
          break
        handle.write(chunk)
  return dest


def resolve_video_path(video_url: str) -> tuple[Path, Optional[Path]]:
  parsed = urlparse(video_url)
  if parsed.scheme in ("http", "https"):
    tmp_dir = Path(tempfile.mkdtemp(prefix="ai-basketball-"))
    tmp_path = tmp_dir / "input.mp4"
    downloaded = download_video(video_url, tmp_path)
    return downloaded, tmp_dir
  if parsed.scheme == "file":
    return Path(parsed.path), None
  return Path(video_url), None


def score_video(video_url: str, model, threshold: Optional[float] = None) -> tuple[Optional[AiEvent], float]:
  video_path, tmp_dir = resolve_video_path(video_url)
  try:
    video, _, info = read_video(str(video_path), pts_unit="sec")
  finally:
    if tmp_dir:
      try:
        if video_path.exists():
          video_path.unlink()
        tmp_dir.rmdir()
      except Exception:
        pass

  fps = info.get("video_fps") or 30.0
  duration = video.shape[0] / fps if fps else 0.0
  inputs = preprocess_frames(video)
  with torch.no_grad():
    logits = model(inputs)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs))
    score = float(probs[pred_idx])
  raw_label = LABELS[pred_idx] if pred_idx < len(LABELS) else f"class_{pred_idx}"
  mapped_label = map_label_to_event(raw_label)

  if threshold is not None and score < threshold:
    return None, score

  if mapped_label is None:
    return None, score

  event = AiEvent(
    id="evt-0",
    type=mapped_label,
    start=0.0,
    end=float(duration),
    confidence=score,
    score=round(score * 10, 2),
  )
  return event, score


def load_checkpoint_from_env() -> Path:
  return resolve_checkpoint_path()


LABELS = load_labels()
MODEL, NUM_CLASSES = load_model(load_checkpoint_from_env(), num_classes=len(LABELS))
# If checkpoint has more classes than labels, extend with generic names
if NUM_CLASSES > len(LABELS):
  extra = [f"class_{i}" for i in range(len(LABELS), NUM_CLASSES)]
  LABELS = LABELS + extra
JOBS: Dict[str, Dict] = {}
app = FastAPI()


@app.get("/health")
def health():
  return {
    "status": "ok",
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  }


@app.post("/api/ai/analyze")
async def analyze(req: AnalyzeRequest):
  job_id = str(uuid.uuid4())
  JOBS[job_id] = {"status": "queued", "result": None}

  async def run_job():
    start_time = time.time()
    try:
      JOBS[job_id] = {"status": "running", "result": None}
      threshold = req.options.get("threshold") if req.options else None

      event, score = score_video(req.videoUrl, MODEL, threshold=threshold)
      processing_ms = (time.time() - start_time) * 1000.0

      if event is None:
        res = AiResult(
          jobId=job_id,
          status="error",
          progress=100,
          message="No highlight class detected",
          events=[],
          diagnostics=AiDiagnostics(processingMs=processing_ms, avgScore=score),
        )
        JOBS[job_id] = {"status": "error", "result": res}
        return

      res = AiResult(
        jobId=job_id,
        status="done",
        progress=100,
        message="ok",
        events=[event],
        diagnostics=AiDiagnostics(processingMs=processing_ms, avgScore=score),
      )
      JOBS[job_id] = {"status": "done", "result": res}
    except Exception as e:
      err_res = AiResult(jobId=job_id, status="error", progress=100, message=str(e), events=[])
      JOBS[job_id] = {"status": "error", "result": err_res}

  asyncio.create_task(run_job())
  return {"jobId": job_id}


@app.get("/api/ai/result/{job_id}")
async def result(job_id: str):
  job = JOBS.get(job_id)
  if not job:
    return AiResult(jobId=job_id, status="error", progress=100, message="job not found", events=[])
  if job["status"] in ("done", "error"):
    return job["result"]
  return AiResult(jobId=job_id, status="running", progress=50, message="processing", events=[])
