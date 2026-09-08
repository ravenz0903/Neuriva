# 🧠 CerebroASPECTS: Automated Topographic Stroke Scoring System
> **Explainable AI Decision Support for Acute Ischemic Stroke Triage**  
> *Developed for 36-Hour National Hackathon | Aligned with AHA/ASA Guidelines & DAWN/DEFUSE-3 Criteria*

[![Live Demo](https://img.shields.io/badge/Demo-Localhost%3A3000-cyan?style=for-the-badge)](http://localhost:3000)
[![FastAPI Backend](https://img.shields.io/badge/Backend-FastAPI%20%3A8000-009688?style=for-the-badge&logo=fastapi)](http://localhost:8000/health)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 🚨 The Clinical Problem
In acute ischemic stroke, **1.9 million neurons die every minute** (*"Time is Brain"*). Emergency triage relies on the **10-point ASPECTS (Alberta Stroke Program Early CT Score)** on non-contrast CT (NCCT) to evaluate candidate eligibility for mechanical thrombectomy and thrombolysis.

However, manual visual scoring suffers from:
- **20–30% Inter-Rater Variability:** Significant disagreement between on-call physicians under acute emergency room pressure.
- **The "Black-Box" AI Barrier:** Clinicians reject uninterpretable heatmaps or raw Dice scores that fail to specify *which* anatomical territory is lost.

---

## ⚡ The CerebroASPECTS Solution
CerebroASPECTS replaces opaque predictions with a **fully explainable, standardized geometric scoring engine** and an **Interactive Anatomical Mind Map**:

```
                               ┌── Caudate (C) [Intact]
             ┌── Subcortical ──┼── Lentiform (L) [Infarcted -1 pt]
             │   (Deep Core)   └── Internal Capsule (IC) [Infarcted -1 pt]
[ ASPECTS: 8/10 ]
             │                 ┌── Insular Ribbon (I) [Intact]
             └── Cortical ─────┼── M1 (Anterior MCA) [Intact]
                 Convexity     ├── M2 (Lateral MCA) [Intact]
                               └── M3 (Posterior MCA) [Intact]
```

### Key Technical Innovations
1. **Unified Dual-Otsu & Silhouette Midline Registration:**
   - **Dual-Otsu Segmentation:** Strips background air and dense cortical bone to isolate pure brain parenchyma.
   - **Silhouette-Based Midline Registration:** Dynamically registers the interhemispheric fissure via silhouette symmetry search (invariant to head tilt or horizontal shifting).
   - **Contralateral Attenuation Differencing:** Explores subtle hypodensity across mirrored hemispheres with an enforced noise floor and hole closure.
   - **Radial Ribbon Contouring:** Sweeps cortical territories (M1–M3) along the measured radial parenchymal boundary.
2. **Clinical DICOM Pipeline:**
   - True 16-bit DICOM ingestion with `RescaleSlope`/`RescaleIntercept` conversion to true Hounsfield Units ($HU = 	ext{Pixel} 	imes 	ext{Slope} + 	ext{Intercept}$).
   - Standard stroke CT windowing ($WW=80, WL=40 \implies [0, 80	ext{ HU}]$).
3. **Interactive Anatomical Mind Map:**
   - SVG-based hierarchical tree with hover-expanding clinical cards showing tissue loss %, estimated HU attenuation, and functional impact.
4. **Clinical Decision Guardrails:**
   - Non-prescriptive advisory card framed around DAWN / DEFUSE-3 trial window context.

---

## 📁 Repository Structure
```text
├── website/
│   └── index.html             # CerebroASPECTS v2 interactive diagnostic workstation
├── server.py                  # FastAPI REST API (http://localhost:8000)
├── vision_engine.py           # Unified dual-Otsu, silhouette midline & ASPECTS scoring core
├── model_service.py           # Pretrained U-Net loader with weights_only=True & fallback
├── dicom_processor.py         # 16-bit DICOM decoding, HU calibration & stroke windowing
├── test_backend.py            # Comprehensive unit & regression verification suite
├── requirements.txt           # Production backend dependencies
├── app.py                     # Streamlit frontend alternative
├── generate_dummy_data.py     # Synthetic NCCT slice & stroke mask generator
├── tune_polygons.py           # Polygon contour calibration visualizer
├── quick_inspect.py           # Checkpoint diagnostic tool
├── BACKEND_TASKS.md           # Backend sprint tasks & architecture reference
├── FRONTEND_TASKS.md          # Frontend integration sprint tasks
└── README.md
```

---

## 🚀 Quick Start

### 1. Launch the Backend API Service
```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server on port 8000
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
Test backend health: [http://localhost:8000/health](http://localhost:8000/health)

### 2. Run Comprehensive Verification Suite
```bash
python test_backend.py
```

### 3. Launch the Diagnostic Web Console
```bash
python -m http.server 3000 --directory website
```
Open [http://localhost:3000](http://localhost:3000) in any modern browser.

---

## 📚 Scientific Literature & Evidence Base
- **ASPECTS-281 NCCT Atlas (Nature Scientific Data, Oct 2024):** [s41597-024-03973-y](https://www.nature.com/articles/s41597-024-03973-y) | Age-stratified NCCT template across 4 cohorts (10–89 yrs).
- **Clinical Validation on MR CLEAN Registry (PMC7966210):** Automated ASPECTS software matching expert neuroradiologist consensus (ICC ~0.71).
- **Symmetry-Enhanced Attention Network (arXiv:2110.05039 / MICCAI):** Bilateral symmetry disentanglement for NCCT stroke segmentation.
- **APIS Paired CT-MRI Dataset (arXiv:2309.15243 / IEEE ISBI):** Paired NCCT and DWI/ADC acute stroke dataset.

---

## ⚖️ Clinical Disclaimer
*CerebroASPECTS is an observational clinical decision support prototype developed during a 36-hour hackathon. It does not provide autonomous medical diagnosis or treatment triage. All scores and contours require verification by a board-certified radiologist.*
