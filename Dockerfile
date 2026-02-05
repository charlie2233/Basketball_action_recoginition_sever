FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.server.txt .
RUN pip install --no-cache-dir -r requirements.server.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

COPY ai_server.py .
COPY dataset/labels_dict.json dataset/labels_dict.json

EXPOSE 8080

CMD ["bash", "-lc", "uvicorn ai_server:app --host 0.0.0.0 --port ${PORT:-8080}"]
