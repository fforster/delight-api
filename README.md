# DELIGHT API

REST API for host galaxy identification of transient candidates using
[DELIGHT](https://github.com/fforster/DELIGHT) (Förster et al. 2022).

Given transient coordinates (RA, Dec), the API downloads a PanSTARRS
image, builds multi-resolution representations, and runs the DELIGHT CNN
to predict the host galaxy position.  It returns 8 predicted positions
(one per rotation/flip of the input), their mean, and the scatter.

## Requirements

- Docker (recommended), **or**
- Python 3.9.1 with a virtual environment

## Quick start with Docker

```bash
# 1. Build the image
docker build -t delight-api .

# 2. Run the container (set your own API key)
docker run -d \
  --name delight-api \
  -p 8000:8000 \
  -e DELIGHT_API_KEY="your-secret-key-here" \
  delight-api

# 3. Test the health endpoint
curl http://localhost:8000/health

# 4. Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key-here" \
  -d '{"oid": "ZTF21example", "ra": 185.7289, "dec": 15.8235}'
```

## Quick start without Docker

```bash
# 1. Install Python 3.9.1 (e.g. via pyenv)
pyenv install 3.9.1
pyenv local 3.9.1

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install DELIGHT from source
git clone https://github.com/fforster/DELIGHT.git /tmp/DELIGHT
cd /tmp/DELIGHT && pip install . && cd -

# 5. Copy the model weights
cp /tmp/DELIGHT/delight/delight/DELIGHT_v1.h5 ./DELIGHT_v1.h5

# 6. Set environment variables
export DELIGHT_API_KEY="your-secret-key-here"
export DELIGHT_MODEL_PATH="./DELIGHT_v1.h5"

# 7. Run the server
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

## API reference

### `GET /health`

Returns `{"status": "ok"}` if the model is loaded.

### `POST /predict`

**Headers:**

| Header      | Value               |
|-------------|---------------------|
| Content-Type | application/json   |
| X-API-Key    | your API key       |

**Request body:**

```json
{
  "oid": "ZTF21example",
  "ra": 185.7289,
  "dec": 15.8235
}
```

| Field | Type   | Description                          |
|-------|--------|--------------------------------------|
| oid   | string | Object identifier                    |
| ra    | float  | Right ascension in degrees [0, 360]  |
| dec   | float  | Declination in degrees [-90, 90]     |

**Response (200):**

```json
{
  "oid": "ZTF21example",
  "ra_input": 185.7289,
  "dec_input": 15.8235,
  "host_predictions": [
    {"ra": 185.7301, "dec": 15.8228},
    {"ra": 185.7299, "dec": 15.8230},
    ...
  ],
  "ra_mean": 185.7300,
  "dec_mean": 15.8229,
  "std_pixels": 1.23
}
```

| Field            | Type           | Description                                         |
|------------------|----------------|-----------------------------------------------------|
| host_predictions | array of 8     | One (RA, Dec) pair per rotation/flip                |
| ra_mean          | float          | Mean predicted host RA (degrees)                    |
| dec_mean         | float          | Mean predicted host Dec (degrees)                   |
| std_pixels       | float          | RMS scatter of the 8 predictions (pixels)           |

**Error responses:**

| Code | Meaning                                               |
|------|-------------------------------------------------------|
| 401  | Missing or invalid API key                            |
| 422  | PanSTARRS image unavailable (outside footprint, etc.) |
| 500  | Internal pipeline failure                             |

## Configuration

All configuration is via environment variables:

| Variable             | Default          | Description                       |
|----------------------|------------------|-----------------------------------|
| `DELIGHT_API_KEY`    | `changeme`       | API key for authentication        |
| `DELIGHT_MODEL_PATH` | `./DELIGHT_v1.h5` | Path to the trained model weights |
| `LOG_LEVEL`          | `INFO`           | Python logging level              |

## Architecture notes

- The TensorFlow model is loaded **once** at startup and reused across
  requests.  This avoids the ~5-10 second initialization per call.
- Each request creates a temporary working directory for the downloaded
  PanSTARRS FITS file, which is cleaned up after the response is sent.
- The server runs with **1 worker** because TensorFlow is not fork-safe.
  For concurrency, run multiple containers behind a load balancer.
- The dominant latency is the PanSTARRS HiPS2fits image download
  (typically 2-10 seconds), not the CNN inference (milliseconds).

## Python example client

```python
import requests

API_URL = "http://localhost:8000/predict"
API_KEY = "your-secret-key-here"

response = requests.post(
    API_URL,
    headers={"X-API-Key": API_KEY},
    json={"oid": "ZTF21example", "ra": 185.7289, "dec": 15.8235},
)
response.raise_for_status()
result = response.json()

for i, pred in enumerate(result["host_predictions"]):
    print(f"  Prediction {i}: RA={pred['ra']:.6f}, Dec={pred['dec']:.6f}")

print(f"  Mean: RA={result['ra_mean']:.6f}, Dec={result['dec_mean']:.6f}")
print(f"  Scatter: {result['std_pixels']:.2f} pixels")
```

## Citation

If you use this service, please cite the DELIGHT paper:

> Förster et al. (2022), "DELIGHT: Deep Learning Identification of Galaxy
> Hosts of Transients using Multiresolution Images", AJ, 164, 195.
> [doi:10.3847/1538-3881/ac912a](https://doi.org/10.3847/1538-3881/ac912a)
