# ⚙️ BACKEND TASK LIST — Person 2 (You)
**Focus:** Python API Server, Real DICOM/Image Pipeline, Model Inference & Telemetry
**Goal:** Build a robust, lightweight FastAPI service on `http://localhost:8000` that connects to the frontend and prepares for Person A's model.

---

## 🛠️ Tech Stack & Setup
- **Framework:** FastAPI + Uvicorn (Fast, async, auto Swagger docs at `/docs`)
- **Imaging Libraries:** `numpy`, `opencv-python`, `pydicom`, `torch` (optional for Person A model)
- **CORS:** Must allow `http://localhost:3000` (and `*`)

---

## 📋 Granular Backend Tasks

### Phase 1: FastAPI Foundation & CORS Setup (Hours 0:00 – 1:00)
- [ ] **Task B1.1: Initialize Backend Environment**
  - Ensure packages are installed: `pip install fastapi uvicorn python-multipart pydicom opencv-python numpy`
  - Create `server.py` with FastAPI instance and `CORSMiddleware` configured to allow all origins/headers/methods.
- [ ] **Task B1.2: Health Check & Metadata Endpoint**
  - Implement `GET /health` returning:
    `{"status": "healthy", "engine": "CerebroASPECTS-v2.0", "device": "cpu" | "cuda", "latency_baseline_ms": 25}`
  - Implement `GET /atlas/info` returning citation info for Nature 2024 ASPECTS-281 atlas.

### Phase 2: Core Ingestion & Analysis Endpoint (Hours 1:00 – 2:30)
- [ ] **Task B2.1: Implement `POST /api/analyze` Endpoint**
  - Input: Form data with:
    * `file`: `UploadFile` (supports `.png`, `.jpg`, `.jpeg`, `.dcm`)
    * `hemisphere`: `str` (`"right"` or `"left"`)
    * `sensitivity`: `float` (default `0.05`)
- [ ] **Task B2.2: True DICOM (.dcm) Decoder with Stroke Windowing**
  - If uploaded file is `.dcm`:
    * Parse with `pydicom.dcmread()`.
    * Extract `RescaleSlope` and `RescaleIntercept` to calculate true Hounsfield Units: $	ext{HU} = 	ext{PixelValue} 	imes 	ext{Slope} + 	ext{Intercept}$.
    * Apply standard stroke window: Window Width ($WW = 80$), Window Level ($WL = 40$).
    * Normalize to 8-bit grayscale array for vision engine.
  - If standard image (`.png`/`.jpg`), decode with `cv2.imdecode`.
- [ ] **Task B2.3: Integrate Scoring Logic (`aspects_engine.py`)**
  - Call `compute_aspects(scan_gray, lesion_mask, affected_hemisphere)`.
  - Package output into clean JSON schema matching frontend contract.

### Phase 3: Model Bridge & Fallback Architecture (Hours 2:30 – 3:30)
- [ ] **Task B3.1: Plug-in Architecture for Person A**
  - Create `model_wrapper.py`:
    * If `models/best_unet.pt` exists and loads cleanly -> run PyTorch inference to produce `lesion_mask`.
    * If model fails or file missing -> fall back automatically to the contralateral symmetry difference algorithm without crashing!
- [ ] **Task B3.2: Export Response Payloads**
  - Encode visual overlay and lesion mask into `base64` strings so the frontend can display them directly:
    ```json
    {
      "aspects_score": 8,
      "hemisphere": "right",
      "telemetry": {
        "parenchyma_px": 118745,
        "midline_x": 255,
        "lesion_px": 4223,
        "inference_ms": 34
      },
      "regions": {
        "C": {"status": "INTACT", "overlap_ratio": 0.0, "mean_hu": 36.2},
        "L": {"status": "INFARCTED", "overlap_ratio": 42.1, "mean_hu": 23.8},
        "IC": {"status": "INFARCTED", "overlap_ratio": 28.4, "mean_hu": 24.1},
        "I": {"status": "INTACT", "overlap_ratio": 0.0, "mean_hu": 35.8},
        "M1": {"status": "INTACT", "overlap_ratio": 0.0, "mean_hu": 36.0},
        "M2": {"status": "INTACT", "overlap_ratio": 0.0, "mean_hu": 35.5},
        "M3": {"status": "INTACT", "overlap_ratio": 0.0, "mean_hu": 36.1}
      },
      "overlay_base64": "data:image/png;base64,...",
      "clinical_summary": "Pattern consistent with favorable core volume (< 70 mL). Eligible for EVT evaluation under DEFUSE-3 window; requires radiologist verification."
    }
    ```

### Phase 4: Integration Smoke Test (Hours 3:30 – 4:30)
- [ ] **Task B4.1: End-to-End Curl / Postman Tests**
  - Test `/api/analyze` using curl with sample slice and verify response is valid JSON under 100ms.
- [ ] **Task B4.2: Coordinate with Person 3**
  - Confirm Person 3 can query `http://localhost:8000/api/analyze` from the frontend.
