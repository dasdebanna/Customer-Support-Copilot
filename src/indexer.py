"""
Index the cleaned corpus into FAISS using sentence-level/small-chunk passages.

Outputs:
 - faiss_index.bin  (FAISS index)
 - docs_meta.jsonl  (one JSON line per vector with fields: id, url, title, text)
"""
from sentence_transformers import SentenceTransformer
import faiss
import ujson as json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re

CORPUS_PATH = Path(__file__).parent.parent.joinpath("docs_corpus.jsonl")
META_PATH = Path(__file__).parent.parent.joinpath("docs_meta.jsonl")
INDEX_PATH = Path(__file__).parent.parent.joinpath("faiss_index.bin")
EMBED_MODEL = "all-MiniLM-L6-v2"


MAX_SENTENCES_PER_CHUNK = 2

SENT_SPLIT_RE = re.compile(r'([.!?])\s+')

def split_into_sentences(text: str):
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    sents = []
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        punct = parts[i+1] if (i+1)<len(parts) else ""
        sent = (chunk + punct).strip()
        if sent:
            sents.append(sent)
    return sents

def build_index():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found at {CORPUS_PATH}. Run src/scrape_docs.py first.")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = []
    meta = []
    idx = 0

    
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading corpus"):
            doc = json.loads(line)
            url = doc.get("url")
            title = doc.get("title","")
            text = doc.get("text","")
            sents = split_into_sentences(text)
            if not sents:
                continue
            
            i = 0
            while i < len(sents):
                chunk_sents = sents[i:i+MAX_SENTENCES_PER_CHUNK]
                chunk_text = " ".join(chunk_sents).strip()
                if chunk_text:
                    meta.append({"id": idx, "url": url, "title": title, "text": chunk_text})
                    idx += 1
                i += MAX_SENTENCES_PER_CHUNK

    if not meta:
        raise RuntimeError("No chunks created from corpus (empty corpus?)")

    
    texts = [m["text"] for m in meta]
    batch_size = 64
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embs.append(embs)
    embeddings = np.vstack(all_embs).astype("float32")

    
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    with META_PATH.open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Built index with {index.ntotal} vectors. Saved to {INDEX_PATH}, meta to {META_PATH}")

if __name__ == "__main__":
    build_index()
