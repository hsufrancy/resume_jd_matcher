import os, json
from openai import OpenAI
import streamlit as st
import numpy as np
from typing import List, Tuple
from .embeddings import embed_text

API_KEY = os.getenv("OPENAI_API_KEY")
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

def semantic_present(keyword: str, chunk_vecs: List[np.ndarray], thresh=0.75) -> bool:
    kv = embed_text(keyword)
    sims = [float(np.dot(kv, cv) / (np.linalg.norm(kv) * np.linalg.norm(cv) + 1e-8)) for cv in chunk_vecs]
    return max(sims) >= thresh if sims else False

def uncover_keywords(cannonical_jd: dict, chunk_vecs: List[np.ndarray]) -> Tuple[List[str], List[str]]:
    # pool all canidate targets from JD JSON
    pool = (
        cannonical_jd.get("skills_programming", []) +
        cannonical_jd.get("skills_ml", []) +
        cannonical_jd.get("skills_data", []) +
        cannonical_jd.get("tools", []) +
        cannonical_jd.get("domain_terms", []) +
        cannonical_jd.get("responsibilities", []) +
        cannonical_jd.get("education", [])
    )

    pool = [p.strip() for p in pool if p.strip()]
    present, missing = [], []
    for kw in sorted(set(pool), key=str.lower):
        (present if semantic_present(kw, chunk_vecs) else missing).append(kw)
        return present, missing