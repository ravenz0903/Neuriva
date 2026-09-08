import sys
from pathlib import Path
_backend_dir = str(Path(__file__).parent.resolve())
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

"""
server.py — CerebroASPECTS FastAPI Backend Service
===================================================
Serves automated ASPECTS scoring on http://localhost:8000.
Integrates unified vision engine with DICOM decoding and pre-trained U-Net.
"""

import time
import base64
import logging
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2

from dicom_processor import process_upload
from model_service import predict_stroke_mask, get_status as model_status, load_model
from vision_engine import analyze_scan_image, synthesize_benchmark_case

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("cerebroaspects.server")

app = FastAPI(
    title="CerebroASPECTS",
    description="Automated ASPECTS scoring for acute ischemic stroke NCCT",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing CerebroASPECTS backend service...")
    try:
        load_model()
    except Exception as e:
        logger.warning(f"Startup model check: {e}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": "CerebroASPECTS v2.1 (Unified Silhouette Vision)",
        "model": model_status(),
    }


def sanitize_for_json(obj):
    """Converts NumPy primitives to standard Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@app.post("/api/analyze")
async def analyze_scan(
    file: UploadFile = File(...),
    hemisphere: str = Form("right"),
):
    """
    Analyzes an uploaded CT scan (DICOM or PNG/JPG) for ASPECTS scoring.
    """
    start_time = time.time()
    hemisphere = hemisphere.lower().strip()
    if hemisphere not in ("left", "right"):
        raise HTTPException(status_code=400, detail="Invalid hemisphere. Must be 'left' or 'right'.")

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    try:
        scan_gray = process_upload(file_bytes, file.filename or "scan.png")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image decoding failed: {e}. Supported formats: DICOM (.dcm), PNG, JPG.")

    # 1. Attempt pre-trained U-Net inference
    model_mask = predict_stroke_mask(scan_gray)

    # 2. Run unified vision engine (with U-Net mask if available, else contralateral symmetry differencing)
    try:
        analysis = analyze_scan_image(scan_gray, hemisphere=hemisphere, external_lesion_mask=model_mask)
    except Exception as e:
        logger.error(f"Analysis engine error: {e}")
        raise HTTPException(status_code=500, detail=f"ASPECTS evaluation failed: {e}")

    # Encode visual overlay as base64 PNG
    overlay_b64 = ""
    if analysis.get("visual_overlay") is not None:
        success, buf = cv2.imencode(".png", analysis["visual_overlay"])
        if success:
            overlay_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    m_info = model_status()

    response = {
        "aspects_score": analysis["aspects_score"],
        "max_score": 10,
        "hemisphere": hemisphere,
        "midline_x": analysis["midline_x"],
        "region_results": sanitize_for_json(analysis["region_results"]),
        "clinical_context": sanitize_for_json(analysis["clinical_context"]),
        "overlay_base64": overlay_b64,
        "telemetry": {
            "processing_time_ms": elapsed_ms,
            "image_dimensions": list(scan_gray.shape),
            "engine": analysis["telemetry"]["engine"],
            "model_status": m_info["model_status"],
            "device": m_info["device"],
            "lesion_pixel_count": analysis["telemetry"]["lesion_pixel_count"],
            "parenchyma_area": analysis["telemetry"]["parenchyma_area"],
            "asymmetry_signal": analysis["telemetry"]["asymmetry_signal"],
        }
    }

    return JSONResponse(content=response)


# ── Benchmark Cohorts (Aligned with Frontend: normal, focal, m1) ──

BENCHMARK_COHORTS = {
    "normal": {
        "id": "normal",
        "case_code": "#0891",
        "name": "Normal Non-Ischemic Baseline",
        "description": "Healthy non-ischemic brain CT — expected ASPECTS 10/10",
        "expected_score": 10,
        "side": "right",
    },
    "focal": {
        "id": "focal",
        "case_code": "#2104",
        "name": "Focal Basal Ganglia Infarct",
        "description": "Deep perforator distribution ischaemia — expected ASPECTS 8-9/10",
        "expected_score_range": [8, 9],
        "side": "right",
    },
    "m1": {
        "id": "m1",
        "case_code": "#4092",
        "name": "Acute M1 Occlusion (Dense MCA Sign)",
        "description": "Extensive MCA cortical and deep territory infarction — expected ASPECTS 3-5/10",
        "expected_score_range": [3, 5],
        "side": "right",
    }
}


@app.get("/api/benchmark/{cohort_id}")
async def get_benchmark(cohort_id: str):
    """
    Returns benchmark cohort metadata along with synthetic image base64.
    """
    if cohort_id == "all":
        items = []
        for cid, meta in BENCHMARK_COHORTS.items():
            item = dict(meta)
            img = synthesize_benchmark_case(cid, side=meta["side"])
            _, buf = cv2.imencode(".png", img)
            item["image_base64"] = base64.b64encode(buf.tobytes()).decode("utf-8")
            items.append(item)
        return {"cohorts": items}

    cohort = BENCHMARK_COHORTS.get(cohort_id)
    if not cohort:
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark cohort '{cohort_id}' not found. Available: {list(BENCHMARK_COHORTS.keys())}, 'all'",
        )

    res = dict(cohort)
    img = synthesize_benchmark_case(cohort_id, side=cohort["side"])
    _, buf = cv2.imencode(".png", img)
    res["image_base64"] = base64.b64encode(buf.tobytes()).decode("utf-8")
    return res


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
