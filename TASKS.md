# 🏁 CEREBROASPECTS: Master Hackathon Roadmap & Task Tracker
**Evaluation Target:** Round 1 (3:00 AM) | **Core Engine Status:** CerebroASPECTS v2 Live (Client-Side Vision Engine)

---

## 🏆 Current Production State (CerebroASPECTS v2 - Verified)
- [x] **Client-Side Real Vision Pipeline:**
  - Dual-Otsu segmentation (air/tissue threshold, then tissue/bone threshold) to cleanly isolate parenchyma.
  - Silhouette-based midline symmetry registration (resolved the 22px lesion-drag bug; midline locked at x=255).
  - Contralateral attenuation differencing across mirrored hemispheres.
  - Connected-component analysis with flood-fill hole closure for confluent stroke contours.
  - Radial sweep cortical ribbon extraction adapting to natural brain boundaries (no hardcoded ellipses).
- [x] **Interactive Anatomical Mind Map:**
  - SVG tree hierarchy: Root -> Subcortical Deep Core (C, L, IC) vs. Cortical Convexity (I, M1-M3).
  - Color-coded node status (Emerald Intact / Crimson Infarcted with pulsing glow filter).
  - Hover-expanding clinical glass cards displaying: anatomical landmark, status chip, tissue loss %, estimated HU attenuation, and clinical notes.
  - Bidirectional cross-highlighting syncing directly between mind map nodes and CT scan contours.
- [x] **Authentic Radiologic HUD & UI:**
  - 4-corner DICOM PACS overlay (`W:80 L:40`, `SLICE: Z=28 (GANGLIONIC)`, `FOV: 220mm`).
  - Real-time 1.5s laser sweep with live measured telemetry log.
  - Dynamic score countdown ticker (10 -> 9 -> ... -> calculated score).
  - Presets tested and validated across Left & Right hemispheres:
    * `#0891 Normal Baseline` -> 10/10 (0 px core)
    * `#2104 Focal Basal Ganglia` -> 8/10 (L, IC involved, 4,223 px core)
    * `#4092 Acute M1 Occlusion` -> 4/10 (L, IC, I, M1-M3 involved, 38,156 px core)

---

## 📚 Integrated Scientific & Clinical Resources (Literature Base)
These peer-reviewed resources and public atlases back our methodology and pitch defense:

1. **ASPECTS-281 NCCT Atlas (Nature Scientific Data 2024 / Figshare / BravoSun):**
   - *Citation:* Sun et al., Nature Scientific Data (Oct 2024) [s41597-024-03973-y](https://www.nature.com/articles/s41597-024-03973-y), [GitHub: BravoSun](https://github.com/BravoSun/NCCT-atlas-for-ASPECTS-scoring), [Figshare: 26819290](https://figshare.com/articles/figure/ASPECTS-281/26819290).
   - *Takeaway:* Age-stratified (10-29, 30-49, 50-69, 70-89) NCCT templates from 281 clinical subjects. Validates our Phase 2 roadmap to account for ventricular enlargement in elderly stroke patients.
2. **Clinical Validation of Automated ASPECTS (PMC7966210 - MR CLEAN Trial Registry):**
   - *Citation:* [PMC7966210](https://pmc.ncbi.nlm.nih.gov/articles/PMC7966210/)
   - *Takeaway:* Validates automated ASPECTS software against human expert consensus; automated scoring achieves comparable reliability (ICC ~0.71), establishing clinical precedent.
3. **APIS Paired CT-MRI Dataset (arXiv:2309.15243 - ISBI 2023):**
   - *Citation:* [arXiv:2309.15243](https://arxiv.org/abs/2309.15243)
   - *Takeaway:* First public dataset of paired NCCT and DWI/ADC MRI for stroke. Confirms early NCCT hypodensity is subtle (2-5 HU), justifying our contralateral differencing pipeline.
4. **Symmetry-Enhanced Attention Network (SEAN / ADN - arXiv:2110.05039 / MICCAI 2022):**
   - *Citation:* [arXiv:2110.05039](https://arxiv.org/abs/2110.05039)
   - *Takeaway:* Uses bilateral symmetry disentanglement to capture ischemic context across hemispheres. Scientifically backs our client-side contralateral differencing approach.
5. **MIPLAB Elderly NCCT & FLAIR Atlas (deepthirajashekar GitHub):**
   - *Citation:* [GitHub: deepthirajashekar](https://github.com/deepthirajashekar/FLAIR-and-NCCT-atlas-for-elderly)
   - *Takeaway:* Standardized NCCT stereotaxic space compatible with MNI ICBM152.
6. **Stroke Population-Specific CT-MRI Atlas (PMC9271109):**
   - *Citation:* [PMC9271109](https://pmc.ncbi.nlm.nih.gov/articles/PMC9271109/)
   - *Takeaway:* Clinical normalization strategies for thick-slice (5mm) emergency CT.
7. **ISLES 2022 Multicenter Benchmark (PMC9741583):**
   - *Citation:* [PMC9741583](https://pmc.ncbi.nlm.nih.gov/articles/PMC9741583/)
   - *Takeaway:* Benchmark standards for ischemic core volume quantification (<70 mL EVT threshold).

---

## 📋 Active Tasks for Final Sprint (Towards 3:00 AM Evaluation)

### Phase 4: Pitch Deck, Demonstration & Defense Readiness
- [ ] **Task 4.1: Slide Deck & Narrative Script (`PITCH_DECK.md`)**
  - Synthesize the 4-slide presentation:
    * *Slide 1 — The Emergency:* "Time is Brain" (1.9M neurons/min) + MR CLEAN validation evidence on inter-rater variability (PMC7966210).
    * *Slide 2 — Explainable Topographic AI:* Contrast raw black-box heatmaps with our 7-region Anatomical Mind Map.
    * *Slide 3 — The Live Client-Side Vision Engine:* Real-time dual Otsu, silhouette midline registration, and contralateral attenuation differencing.
    * *Slide 4 — Validated Scientific Roadmap:* Citing the 2024 Nature ASPECTS-281 atlas and APIS paired CT-MRI dataset for Phase 2 deformable registration.
- [ ] **Task 4.2: Judge Defense Matrix Rehearsal**
  - Prepare bulletproof verbal answers for:
    * *Q1 (DICOM handling):* Honest acknowledgement that browser v1 decodes PNG/JPEG raster slices, while Phase 2 ingests raw 16-bit DICOM Hounsfield scaling.
    * *Q2 (Registration vs. Heuristic):* Citing the Nature 2024 ASPECTS-281 paper as our exact Phase 2 template integration.
    * *Q3 (Clinical Authority):* Reinforce non-prescriptive decision support under DAWN / DEFUSE-3 trial contexts.
- [ ] **Task 4.3: Backup Screen Recording**
  - Record a 60-second walkthrough video of the live site in action (laser sweep, benchmark switching, mind map hover) to ensure zero demo-day risks.

### Phase 5: Person A Integration (Optional / Parallel)
- [ ] **Task 5.1: Model Weights Checkpoint Hook**
  - If Person A provides working inference weights from `herutriana44/acute-ischemic-stroke-unet`, wrap into `app.py` or export output mask into the web pipeline.
  - If Person A is delayed, the client-side vision engine in `website/index.html` serves as a 100% self-contained, working product.
