import os, json, textwrap
from openai import OpenAI
import streamlit as st
from typing import List, Dict

API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=API_KEY)

def grade_resume(canonical_jd: dict, top_chunks: List[str], model:str ="gpt-4o-mini") -> Dict:
    system = """
    You are a hiring evaluator. You must grade resumes on a calibrated 0-100 scale.

    Use THIS scoring standard:
    - 90-100 = Excellent match; resume strongly aligns with most JD requirements.
    - 80-89 = Good match; candidate meets many key requirements.
    - 70-79 = Decent match; candidate meets some requirements but needs improvements.
    - 60-69 = Weak match; noticeable gaps exist but candidate is not hopeless.
    - below 60 = Poor match; major gaps.

    Do NOT be overly strict. Most realistic applicants score between 65-90.
    Return strict JSON only.
    """
    jd_json = json.dumps(canonical_jd, ensure_ascii=False)
    chunks_preview = "\n\n---\n\n".join([c[:1200] for c in top_chunks])  # limit tokens
    user = f"""
    Evaluate the candidate using this rubric.

    Canonical JD (JSON):
    {jd_json}

    Candidate Resume Evidence (top chunks):
    {chunks_preview}

    Return JSON:
    {{
      "overall_score": <0-100 integer using the calibrated scale>,
      "subscores": {{
          "skills": <0-100>,
          "tools": <0-100>,
          "responsibilities": <0-100>,
          "domain": <0-100>,
          "education": <0-100>,
          "seniority": <0-100>
      }},
      "missing_keywords": ["keyword1", "keyword2", ...]
    }}
    Output JSON ONLY.
    """


    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}]
    )
    data = resp.choices[0].message.content
    return json.loads(data)