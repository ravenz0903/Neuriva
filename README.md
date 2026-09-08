# 🧠 CerebroASPECTS: Automated Topographic Stroke Scoring System
> **Explainable AI Decision Support for Acute Ischemic Stroke Triage**  
> *Developed for 36-Hour National Hackathon | Aligned with AHA/ASA Guidelines & DAWN/DEFUSE-3 Criteria*

[![Live Demo](https://img.shields.io/badge/Demo-Localhost%3A3000-cyan?style=for-the-badge)](http://localhost:3000)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
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
1. **Real-Time Client-Side Vision Engine:**
   - **Dual-Otsu Segmentation:** Strips background air and dense cortical bone to isolate pure brain parenchyma.
   - **Silhouette-Based Midline Registration:** Locks the interhemispheric fissure to x=255 without lesion-drag distortion.
   - **Contralateral Attenuation Differencing:** Explores subtle hypodensity (23 HU edema vs. 36 HU healthy tissue) across mirrored hemispheres.
   - **Radial Ribbon Contouring:** Dynamically sweeps cortical territories (M1–M3) along the measured radial contour of the segmented brain rather than a hardcoded ellipse.
2. **Interactive Anatomical Mind Map:**
   - SVG-based hierarchical tree with hover-expanding clinical cards showing tissue loss %, estimated HU attenuation, and functional clinical impact.
   - Bidirectional cross-highlighting syncing directly between mind map nodes and CT scan contours.
3. **Clinical Guardrails:**
   - Non-prescriptive advisory card framed around DAWN / DEFUSE-3 trial window context.
   - 4-corner DICOM PACS HUD overlay (W:80, L:40, Z=28).

---

## 📁 Repository Structure
```text
├── website/
│   └── index.html             # CerebroASPECTS v2 interactive diagnostic workstation
├── aspects_engine.py          # Python geometric normalization & ASPECTS scoring core
├── app.py                     # Streamlit frontend alternative
├── generate_dummy_data.py     # Generates synthetic NCCT slices & stroke masks
├── tune_polygons.py           # Polygon contour alignment & visualizer
├── quick_inspect.py           # Pretrained U-Net weights diagnostic utility
├── TASKS.md                   # Master hackathon roadmap & literature tracker
├── BACKEND_TASKS.md           # Person 2 FastAPI backend sprint tasks
├── FRONTEND_TASKS.md          # Person 3 frontend integration sprint tasks
└── README.md
```

---

## 🚀 Quick Start

### 1. Launch the Diagnostic Web Console
```bash
# From the repository root:
python -m http.server 3000 --directory website
```
Open [http://localhost:3000](http://localhost:3000) in any modern web browser.

### 2. Run the Python Geometric Engine
```bash
python -m venv .venv
# Activate: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux/Mac)
pip install numpy opencv-python matplotlib streamlit
python aspects_engine.py
```

### 3. Generate Synthetic Benchmark Datasets
```bash
python generate_dummy_data.py
```

---

## 📚 Scientific Literature & Evidence Base
- **ASPECTS-281 NCCT Atlas (Nature Scientific Data, Oct 2024):** [s41597-024-03973-y](https://www.nature.com/articles/s41597-024-03973-y) | Age-stratified NCCT template across 4 cohorts (10–89 yrs).
- **Clinical Validation on MR CLEAN Registry (PMC7966210):** Validates automated ASPECTS software matching expert neuroradiologist consensus (ICC ~0.71).
- **Symmetry-Enhanced Attention Network (arXiv:2110.05039 / MICCAI):** Scientifically validates bilateral symmetry disentanglement for NCCT stroke segmentation.
- **APIS Paired CT-MRI Dataset (arXiv:2309.15243 / IEEE ISBI):** First paired NCCT and DWI/ADC acute stroke dataset.

---

## ⚖️ Clinical Disclaimer
*CerebroASPECTS is an observational clinical decision support and research prototype developed during a 36-hour hackathon. It does not provide autonomous medical diagnosis or treatment triage. All scores and contours require verification by a board-certified radiologist.*
