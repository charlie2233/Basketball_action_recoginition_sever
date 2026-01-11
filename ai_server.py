"""
Minimal FastAPI inference server for the basketball action model.

Endpoints (matches the mobile app expectation):
  POST /api/ai/analyze { "videoUrl": "<path-or-url>" } -> { "jobId": "<uuid>" }
  GET  /api/ai/result/{jobId} -> AiResult with events and diagnostics

Usage:
  uvicorn ai_server:app --host 0.0.0.0 --port 8787
    (optional) set CHECKPOINT env var to override default checkpoint path.

Notes:
  - Default checkpoint: model_checkpoints/r2plus1d_clean/r2plus1d_multiclass_13_0.0001.pt
  - Designed for local/edge inference: single-event per clip, covering full clip duration.
  - For remote URLs, download to a temp file before scoring or mount storage accordingly.
"""

import asyncio
import os
import uuid
from pathlib import Path
from typing import Dict, Optional

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

LABELS = ['dunk', 'three_pointer', 'made_shot', 'block', 'steal', 'assist', 'foul']  # fallback; will be extended if checkpoint has more classes
DEFAULT_CKPT = Path("model_checkpoints/r2plus1d_clean/r2plus1d_multiclass_13_0.0001.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def score_video(video_path: str, model) -> tuple[AiEvent, float]:
  video, _, info = read_video(video_path, pts_unit="sec")
  fps = info.get("video_fps") or 30.0
  duration = video.shape[0] / fps if fps else 0.0
  inputs = preprocess_frames(video)
  with torch.no_grad():
    logits = model(inputs)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs))
    score = float(probs[pred_idx])
  label = LABELS[pred_idx] if pred_idx < len(LABELS) else f"class_{pred_idx}"
  event = AiEvent(
    id="evt-0",
    type=label,
    start=0.0,
    end=float(duration),
    confidence=score,
    score=score,
  )
  return event, score


def load_checkpoint_from_env() -> Path:
  override = os.environ.get("CHECKPOINT")
  if override:
    return Path(override)
  return DEFAULT_CKPT


MODEL, NUM_CLASSES = load_model(load_checkpoint_from_env(), num_classes=len(LABELS))
# If checkpoint has more classes than default labels, extend with generic names
if NUM_CLASSES > len(LABELS):
  extra = [f"class_{i}" for i in range(len(LABELS), NUM_CLASSES)]
  LABELS = LABELS + extra
JOBS: Dict[str, Dict] = {}
app = FastAPI()


@app.post("/api/ai/analyze")
async def analyze(req: AnalyzeRequest):
  job_id = str(uuid.uuid4())
  JOBS[job_id] = {"status": "queued", "result": None}

  async def run_job():
    try:
      event, score = score_video(req.videoUrl, MODEL)
      res = AiResult(
        jobId=job_id,
        status="done",
        progress=100,
        message="ok",
        events=[event],
        diagnostics=AiDiagnostics(processingMs=0.0, avgScore=score),
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
  if job["status"] != "done":
    return AiResult(jobId=job_id, status="running", progress=50, message="processing", events=[])
  return job["result"]
