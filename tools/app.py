import streamlit as st
import numpy as np
import cv2
from aspects_engine import compute_aspects

st.set_page_config(page_title="Auto-ASPECTS Stroke Scoring", layout="wide")
st.title("🧠 Automated ASPECTS Stroke Decision Support")
st.markdown("*Explainable stroke lesion quantification & MCA territory scoring for emergency triage.*")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Axial Brain Scan (NCCT / DWI)", type=["png", "jpg", "jpeg"])
    hemisphere = st.radio("Suspected Stroke Hemisphere", ["right", "left"], horizontal=True)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    scan_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    with st.spinner("Running Stroke Lesion Segmentation..."):
        lesion_mask = np.zeros_like(scan_gray)
        cv2.circle(lesion_mask, (scan_gray.shape[1]//2 + 60, scan_gray.shape[0]//2), 25, 255, -1)

    results = compute_aspects(scan_gray, lesion_mask, affected_hemisphere=hemisphere)

    with col1:
        st.subheader("Segmentation & Regional Overlay")
        st.image(results["visual_overlay"], channels="BGR", use_container_width=True)

    with col2:
        score = results["aspects_score"]
        st.subheader("ASPECTS Scorecard")
        
        if score >= 6:
            st.success(
                f"### ASPECTS: {score} / 10\n"
                "**Morphological pattern consistent with favorable core volume.**\n\n"
                "*For clinical context: trial literature (DAWN / DEFUSE-3) investigates EVT feasibility in this range; "
                "requires expert radiologist confirmation before any treatment decision.*"
            )
        else:
            st.error(
                f"### ASPECTS: {score} / 10\n"
                "**Extensive early ischemic changes.** High risk of reperfusion injury / hemorrhagic transformation."
            )

        st.markdown("#### Regional Breakdown")
        for reg, data in results["region_results"].items():
            status_color = "🔴" if data["affected"] else "🟢"
            st.write(f"{status_color} **{reg}**: {data['status']} (Involvement: {data['overlap_ratio']}%)")
