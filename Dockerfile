FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Dependency layer
# -----------------------------

COPY requirements-docker.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
        python -m pip install --upgrade \
        pip setuptools wheel \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch \
    && python -m pip install \
        -r requirements-docker.txt

# -----------------------------
# Application layer
# -----------------------------

COPY model ./model
COPY app.py .
COPY lib ./lib
COPY html ./html


EXPOSE 8000

CMD ["python",    "-m",    "uvicorn",    "app:app",    "--host",    "0.0.0.0",    "--port",    "8000" ]