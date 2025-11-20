from pypdf import PdfReader
import re

def _clean_text(t: str) -> str:
    t = t.replace("\x00", "").replace("\r", " ").replace("\t", " ")
    t = re.sub(r"-\n", "", t) # de-hyphenatev
    t = re.sub(r"\n+", "\n", t) # collapse blank lines
    t = re.sub(r"[\t]+", " ", t) # collapse spaces
    return t.strip() # remove leading/trailing whitespace

def pdf_to_text(file) -> str:
    reader = PdfReader(file)
    pages = []
    for p in reader.pages: # p: each page
        pages.append(p.extract_text() or "")
    return _clean_text("\n".join(pages))

def pdfs_to_text(files) -> dict:
    return {f.name: pdf_to_text(f) for f in files}