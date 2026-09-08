# 🎨 FRONTEND TASK LIST — Person 3 (New Teammate)
**Focus:** UI/UX Polish, Backend API Integration, Clinical Export & User Flow
**Current Codebase:** `website/index.html` (v2 — 2,242 lines, running on `http://localhost:3000`)
**Architecture:** Single self-contained HTML5 file with Tailwind CSS CDN.

---

## 🚀 Onboarding Briefing for Person 3
- The core client-side vision pipeline and interactive SVG Anatomical Mind Map are **already built and verified**.
- The backend API on `http://localhost:8000` has now been upgraded to the exact same unified silhouette-based scoring engine (`vision_engine.py`). Switching between backend and browser engine produces 100% consistent scores and classifications.
- The web console runs at `http://localhost:3000`, and FastAPI backend runs at `http://localhost:8000`.

---

## 📋 Granular Frontend Tasks

### Phase 1: Connect Frontend to Person 2's Backend API (Hours 0:00 – 1:30)
- [ ] **Task F1.1: Backend Connection Switch**
  - In `website/index.html`, locate the `CONFIG` object:
    ```javascript
    const CONFIG = {
      BACKEND_URL: "http://localhost:8000/api/analyze",
      HEALTH_URL: "http://localhost:8000/health",
      USE_MOCK: false
    };
    ```
  - On page load, execute a silent check to `HEALTH_URL`:
    * If online: Show green badge in navbar: `[● BACKEND ONLINE: FASTAPI]`.
    * If offline: Show cyan badge: `[● BROWSER CLIENT-SIDE ENGINE]`.
- [ ] **Task F1.2: Live API Ingestion Hook**
  - When a user uploads a file through the dropzone:
    * If backend is online: `FormData.append("file", file)` and `fetch(CONFIG.BACKEND_URL)`.
    * If backend is offline: Gracefully fall back to the existing in-browser pipeline so the demo **NEVER fails**.
  - Ingest DICOM (`.dcm`) uploads by routing them to `CONFIG.BACKEND_URL` for 16-bit Hounsfield calibration and stroke windowing!
  - Populate the Mind Map and Scorecard using the backend's JSON response.

### Phase 2: Clinical Polish & User Interaction (Hours 1:30 – 3:00)
- [ ] **Task F2.1: One-Click "Export Clinical Triage Report" (PDF / Print)**
  - Add a button in the scorecard: `[📄 Export Triage Summary]`.
  - When clicked, opens a clean print-ready medical report dialog containing:
    * Patient Telemetry & Scan timestamp.
    * Final ASPECTS Score (e.g. 08/10) & recommendation card.
    * Regional territory breakdown table (C, L, IC, I, M1-M3).
    * Embedded snapshot of the CT overlay.
    * Clinician sign-off signature line.
- [ ] **Task F2.2: Interactive Windowing (W:80 L:40) Slider**
  - In the CT viewer controls, add two range sliders:
    * **Window Width (WW):** Default 80 (range 40-200).
    * **Window Level (WL):** Default 40 (range 10-80).
  - Adjust canvas contrast/brightness in real-time.

### Phase 3: Anatomical Scope & Supraganglionic Cut Toggle (Hours 3:00 – 4:00)
- [ ] **Task F3.1: Add Axial Level Switcher**
  - In the PACS viewer toolbar, add a toggle:
    * `[Level 1: Ganglionic Cut (C, L, IC, I, M1-M3)]` (Active v1)
    * `[Level 2: Supraganglionic Cut (M4, M5, M6)]` (Phase 2 preview badge)
  - Citing the Nature 2024 ASPECTS-281 atlas in a subtle tooltip.

### Phase 4: Mobile / Tablet Responsiveness & Pitch Polish (Hours 4:00 – 4:30)
- [ ] **Task F4.1: Presentation Screen Verification**
  - Verify UI scaling on 1080p and 4K external monitors / projectors.
  - Verify that hovering mind map nodes works smoothly with zero tooltip overflow.
