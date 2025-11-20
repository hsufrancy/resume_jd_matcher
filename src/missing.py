import os, json, textwrap
from typing import List, Dict
from openai import OpenAI
import streamlit as st

API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=API_KEY)

def propose_missing_keywords(canonical_jd: dict, resume_best_chunks: List[str],
                             uncovered: List[str], model: str = "gpt-4o-mini", top_n:int = 10) -> List[str]:
    system = "You propose concise, high-impact resume keywords. Return JSON only."
    jd_json = json.dumps(canonical_jd, ensure_ascii=False)
    chunks = "\n\n---\n\n".join([c[:1000] for c in resume_best_chunks])
    uncovered_str = ", ".join(uncovered[:40])  # cap length
    user = textwrap.dedent(f"""
    Canonical JD:
    {jd_json}

    Candidate Resume Evidence (best chunks):
    {chunks}

    These items are not covered semantically:
    {uncovered_str}

    Suggest ONLY the top {top_n} keywords or short phrases that would most
    increase this candidate's alignment with the JD. Avoid duplicates and fluff.
    Return JSON: {{"missing_keywords": ["kw1","kw2", "..."]}}
    """).strip()

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}]
    )
    data = json.loads(resp.choices[0].message.content)
    return data.get("missing_keywords", [])[:top_n]
