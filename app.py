"""
DELIGHT API — FastAPI service for host galaxy identification.

Wraps the DELIGHT library (Förster et al. 2022) to expose a REST endpoint
that accepts transient coordinates and returns the 8 predicted host galaxy
positions (one per rotation/flip of the input image).
"""

import os
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

# Set matplotlib backend before any other imports that might trigger it
import matplotlib
matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from delight.delight import Delight

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("DELIGHT_API_KEY", "changeme")
MODEL_PATH = os.environ.get(
    "DELIGHT_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "DELIGHT_v1.h5"),
)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("delight-api")

# ---------------------------------------------------------------------------
# Global state: TF model loaded once at startup
# ---------------------------------------------------------------------------

_tf_model: Optional[tf.keras.Model] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the TensorFlow model once when the application starts."""
    global _tf_model
    logger.info("Loading DELIGHT model from %s", MODEL_PATH)
    _tf_model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DELIGHT API",
    description=(
        "Host galaxy identification for transient candidates using "
        "DELIGHT (Förster et al. 2022). Returns 8 predicted host "
        "positions from rotations and flips of multi-resolution "
        "PanSTARRS images."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: Optional[str] = Security(api_key_header)):
    if key is None or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    oid: str = Field(..., description="Object identifier")
    ra: float = Field(..., description="Right ascension in degrees", ge=0, le=360)
    dec: float = Field(..., description="Declination in degrees", ge=-90, le=90)


class HostCoordinate(BaseModel):
    ra: float = Field(..., description="Predicted host RA in degrees")
    dec: float = Field(..., description="Predicted host Dec in degrees")


class PredictResponse(BaseModel):
    oid: str
    ra_input: float
    dec_input: float
    host_predictions: List[HostCoordinate] = Field(
        ..., description="8 predicted host positions (one per rotation/flip)"
    )
    ra_mean: float = Field(..., description="Mean predicted host RA in degrees")
    dec_mean: float = Field(..., description="Mean predicted host Dec in degrees")
    std_pixels: float = Field(
        ..., description="RMS scatter of the 8 predictions in pixels"
    )


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------


def run_delight(oid: str, ra: float, dec: float) -> PredictResponse:
    """
    Run the full DELIGHT pipeline for a single transient:
      1. Download PanSTARRS image via HiPS2fits
      2. Compute WCS pixel coordinates
      3. Build multi-resolution representation
      4. Preprocess and predict with CNN
      5. Convert 8 pixel-offset predictions to sky coordinates
    """

    workdir = tempfile.mkdtemp(prefix="delight_")

    try:
        # Instantiate Delight for this single object
        oids = np.array([oid])
        ras = np.array([ra])
        decs = np.array([dec])

        dclient = Delight(workdir, oids, ras, decs)

        # Step 1: download PanSTARRS image directly (bypass DELIGHT's
        # threaded download, which has a race condition with < 10 objects)
        logger.info("Downloading PanSTARRS image for %s (%.6f, %.6f)", oid, ra, dec)
        dclient.get_PS1_r(ra, dec)
        dclient.check_missing()

        # Verify download succeeded
        if "filename" not in dclient.df.columns or dclient.df.loc[oid, "filename"] == "":
            raise HTTPException(
                status_code=422,
                detail=(
                    f"PanSTARRS image could not be downloaded for "
                    f"({ra}, {dec}). The source may be outside the "
                    f"PanSTARRS footprint."
                ),
            )

        # Step 2: WCS and pixel coordinates
        dclient.get_pix_coords()

        # Step 3: multi-resolution images
        dclient.compute_multiresolution(
            nlevels=5, domask=False, doobject=True, doplot=False
        )

        # Step 4: inject pre-loaded model and predict
        dclient.tfmodel = _tf_model
        dclient.preprocess()
        dclient.predict()

        # Step 5: extract the 8 individual sky coordinates
        row = dclient.df.loc[oid]
        dxdy_all = row["dxdy_delight_rotflip"]  # shape (8, 2)
        wcs = row["wcs"]
        xSN = row["xSN"]
        ySN = row["ySN"]

        host_predictions = []
        for dx, dy in dxdy_all:
            sky = wcs.pixel_to_world(xSN + dx, ySN + dy)
            host_predictions.append(
                HostCoordinate(
                    ra=float(sky.ra.deg),
                    dec=float(sky.dec.deg),
                )
            )

        return PredictResponse(
            oid=oid,
            ra_input=ra,
            dec_input=dec,
            host_predictions=host_predictions,
            ra_mean=float(row["ra_delight"]),
            dec_mean=float(row["dec_delight"]),
            std_pixels=float(row["std_delight"]),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("DELIGHT pipeline failed for %s", oid)
        raise HTTPException(
            status_code=500,
            detail=f"DELIGHT pipeline failed: {exc}",
        )
    finally:
        # Clean up downloaded FITS files
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    _key: str = Depends(verify_api_key),
):
    """
    Predict the host galaxy position for a single transient candidate.

    Accepts an object identifier and equatorial coordinates (J2000).
    Returns 8 predicted host positions from rotations/flips of the
    multi-resolution input image, plus the mean position and scatter.
    """
    return run_delight(request.oid, request.ra, request.dec)


@app.get("/health")
async def health():
    """Health check — returns 200 if the model is loaded."""
    if _tf_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ok"}
