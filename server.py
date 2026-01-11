from fastapi import FastAPI
import torch

app = FastAPI()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

# Later we’ll add: load checkpoint + /predict endpoint
