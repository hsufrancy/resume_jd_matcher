def sliding_windows(text: str, max_chars=6000, overlap_chars=600):
    chunks = []
    i = 0 # start
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        chunk = text[i:end]
        chunks.append(chunk)
        if end == n:
            break
        i = end - overlap_chars
        if i < 0:
            i = 0
    return chunks
