import os, json
from openai import OpenAI
import streamlit as st
import numpy as np
from typing import List, Tuple
from .embeddings import embed_text

API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=API_KEY)

CANON_SCHEMA = {
    "type": "object",
    "properties": {
        "skills_programming": {"type": "array", "items": {"type": "string"}},
        "skills_ml": {"type": "array", "items": {"type": "string"}},
        "skills_data": {"type": "array", "items": {"type": "string"}},
        "tools": {"type": "array", "items": {"type": "string"}},
        "responsibilities": {"type": "array", "items": {"type": "string"}},
        "domain_terms": {"type": "array", "items": {"type": "string"}},
        "education": {"type": "array", "items": {"type": "string"}},
        "seniority": {"type": "array", "items": {"type": "string"}},
        "synonyms": {"type": "object"}                   
    },
    "required": ["skills_programming", "skills_ml", "skills_data", "tools", "responsibilities", "domain_terms", "education", "seniority", "synonyms"]
}

def jd_to_canonical(jd_text: str, model="gpt-4o-mini") -> dict:
    system = "You are an expert at extracting key information from job descriptions. Given a job description, extract the relevant keywords and phrases according to the provided schema. Respond only with a JSON object that adheres to the schema."
    user = f"""Job Description:
{jd_text}

Return JSON with: skills_programming, skills_ml, skills_data, tools, responsibilities, domain_terms, education, seniority (junor|mid|senior), synonyms (mapping canonical -> aliases). No explanations, only JSON."""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                    {"role": "user", "content": user}]
    )
    data = resp.choices[0].message.content
    return json.loads(data)

def _normalize_field_to_list(value) -> List[str]:
    """
    Normalize JD fields to a list of strings.
    - If it's a string: wrap it in a list.
    - If it's a list: keep only non-empty strings.
    - Otherwise: return [].
    """
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if isinstance(v, str) and v.strip()]
    return []


def semantic_present(keyword: str, chunk_vecs: List[np.ndarray], thresh: float = 0.75) -> bool:
    if not chunk_vecs:
        return False

    kv = embed_text(keyword)
    kv_norm = np.linalg.norm(kv) + 1e-8

    sims = []
    for cv in chunk_vecs:
        denom = kv_norm * (np.linalg.norm(cv) + 1e-8)
        sims.append(float(np.dot(kv, cv) / denom))

    return max(sims) >= thresh


def uncover_keywords(
    canonical_jd: dict, chunk_vecs: List[np.ndarray]
) -> Tuple[List[str], List[str]]:
    """
    Build a pool of keywords from the canonical JD and split into:
    - present: semantically covered in the resume
    - missing: not covered
    """
    fields = [
        "skills_programming",
        "skills_ml",
        "skills_data",
        "tools",
        "domain_terms",
        "responsibilities",
        "education",
    ]

    pool: List[str] = []
    for key in fields:
        value = canonical_jd.get(key, [])
        pool.extend(_normalize_field_to_list(value))

    # dedupe & clean
    pool = [p for p in {p.strip() for p in pool if p.strip()}]

    present, missing = [], []
    for kw in sorted(pool, key=str.lower):
        if semantic_present(kw, chunk_vecs):
            present.append(kw)
        else:
            missing.append(kw)

    return present, missing
