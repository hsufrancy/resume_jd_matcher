import os
from openai import OpenAI
import streamlit as st
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)
_MODEL = "text-embedding-3-large" # Get the embedding model

def _normalize(s: str) -> str:
    return " ".join(s.split())

def embed_text(text: str) -> np.ndarray:
    text = _normalize(text)
    emb = client.embeddings.create(model=_MODEL, input=[text]).data[0].embedding
    return np.array(emb, dtype=np.float32)