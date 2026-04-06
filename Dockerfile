FROM python:3.9-slim-bookworm

# System dependencies required by sep (C extension), astropy (ERFA), and
# general build tools.  We install, build, then remove the compiler to
# keep the final image smaller.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ \
        libffi-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Python dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Install DELIGHT from PyPI ----------
RUN pip install --no-cache-dir --no-deps astro-delight && \
    # Copy the model weights to a known location
    cp $(python -c "import pkg_resources; print(pkg_resources.resource_filename('delight.delight', 'DELIGHT_v1.h5'))") /app/DELIGHT_v1.h5

# ---------- Remove build tools (optional, saves ~150 MB) ----------
RUN apt-get purge -y gcc g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# ---------- Application code ----------
COPY app.py .

# ---------- Runtime configuration ----------
ENV DELIGHT_MODEL_PATH=/app/DELIGHT_v1.h5
ENV DELIGHT_API_KEY=""
ENV LOG_LEVEL=INFO

# Expose the API on port 8000
EXPOSE 8000

# Run with uvicorn.  --workers 1 because TensorFlow is not fork-safe
# and we load the model in-process.  Use a reverse proxy (nginx/caddy)
# in front if you need TLS.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
