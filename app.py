import streamlit as st
import numpy as np
from src.pdf_io import pdfs_to_text
from src.embeddings import embed_text
from src.match_utils import cosine
from src.chunker import sliding_windows
from src.keywords import jd_to_canonical, uncover_keywords
from src.grader import grade_resume
from src.missing import propose_missing_keywords

@st.cache_data(show_spinner=False)
def cache_canonical(jd_text):
    return jd_to_canonical(jd_text)

st.set_page_config(page_title="Resume-JD Matcher", layout="wide")

st.sidebar.header("Input")
files = st.sidebar.file_uploader("Upload Resume (PDFs)", type=["pdf"], accept_multiple_files=True)
jd_text = st.sidebar.text_area("Paste Job Description", height=220)
run = st.sidebar.button("Run Matching", type="primary", use_container_width=True)

st.title("Resume & JD Matcher")

if run:
    if not jd_text or not files:
        st.warning("Please upload at least one resume PDF and paste a job description.")
        st.stop()
    
    # 1) Canonicalize JD (LLM)
    canonical_jd = cache_canonical(jd_text)

    # 2) Embed JD once
    jd_vec = embed_text(jd_text) # embedded jd text

    results = []

    for f in files:
        text = pdfs_to_text([f])[f.name]
        if not text.strip():
            st.error(f"{f.name}: could not extract text (maybe scanned PDF).")
            continue

        # 3) Chunk resume text
        chunks = sliding_windows(text)
        chunk_vecs = [embed_text(c) for c in chunks]

        # 4) Compute cosin sims & picl top-k chunks
        sims = [cosine(jd_vec, v) for v in chunk_vecs]
        k = 3
        top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        top_chunks = [chunks[i] for i in top_idx]
        top_tims = [sims[i] for i in top_idx]

        # embedding-based score (0-100)
        embed_score = int(round(100 * max(0.0, sum(top_tims) / len(top_tims)))) # average top-k similarities

        # 5) LLM rubric score
        graded = grade_resume(canonical_jd, top_chunks)
        llm_score = graded['overall_score']

        # 6) Final score (weighted average)
        final_score = int(round(0.8 * llm_score + 0.2 * embed_score))

        # 7) Missing keywords (semantic + LLM)
        present, uncovered = uncover_keywords(canonical_jd, chunk_vecs)
        missing_kw = propose_missing_keywords(canonical_jd, top_chunks, uncovered, top_n=10)

        results.append({
            "name": f.name,
            "score": final_score,
            "missing_keywords": missing_kw
        })

    # 8) Display results
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    for r in results:
        st.write(f"### {r['name']}")
        st.progress(r["score"] / 100, text=f"Match score: {r['score']}/100")
        st.write("Missing keywords:", ", ".join(r["missing_keywords"]) or "(none)")
        st.markdown("---")