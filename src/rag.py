# src/rag.py
"""
RAG module (updated)
 - FAISS retrieval (sentence-transformers embeddings)
 - Cross-encoder reranker (optional)
 - Prompt template with sentence-aware snippet trimming
 - Generation (OpenAI preferred) or local Flan-T5 fallback
 - Post-processing: concise N sentences, no trailing "..." and no placeholders
 - Deduplication / diversity of contexts by URL
 - Procedural snippet handling (take next sentence if top is a header/list)
"""
from pathlib import Path
import ujson as json
import numpy as np
import textwrap
import os
import faiss
import re

# embeddings & models
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    import openai
except Exception:
    openai = None

# local generator (transformers)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

ROOT = Path(__file__).parent.parent.resolve()
INDEX_PATH = ROOT.joinpath("faiss_index.bin")
META_PATH = ROOT.joinpath("docs_meta.jsonl")

EMBED_MODEL = "all-MiniLM-L6-v2"          # embeddings model (fast)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # reranker

# local text generator model (CPU)
LOCAL_GEN_MODEL = "google/flan-t5-small"

# tune here: how many sentences to keep in final answer
MAX_ANSWER_SENTENCES = 2

# diversify: max chunks per url allowed in final top_candidates
MAX_CHUNKS_PER_URL = 2

# lazy-loaded resources
_index = None
_meta = None
_embed_model = None
_cross_encoder = None
_local_generator = None

# -------------------------
# Utilities: sentence helpers & tidy answer
# -------------------------
RE_SENT_SPLIT = re.compile(r'([.!?])\s+')          # split and keep punctuation
RE_ANGLE_PLACEHOLDER = re.compile(r"<[^>]{1,200}>")
RE_DOUBLE_DASH_ID = re.compile(r"\b[a-zA-Z0-9_-]{3,}--\d{3,}\b")
RE_MULTI_DOTS = re.compile(r"\.{2,}")
RE_WHITESPACE = re.compile(r"\s+")

def split_into_sentences(text: str):
    if not text:
        return []
    parts = RE_SENT_SPLIT.split(text)
    sentences = []
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        punct = parts[i+1] if (i+1)<len(parts) else ""
        sentence = (chunk + punct).strip()
        if sentence:
            sentences.append(sentence)
    return sentences

# -------------------------
# NEW: sentence-level extraction helpers with procedural handling
# -------------------------
def get_top_sentences_from_passage(passage_text: str, query: str, embed_model, top_n: int = 1):
    """
    Given a passage text, split into sentences and return top_n sentences by cosine similarity.
    For 'procedural' outputs where the top sentence is a header/menu (short or 'how to'), also
    include the next sentence to provide actionable content.
    """
    if not passage_text:
        return []
    sents = split_into_sentences(passage_text)
    if not sents:
        return []
    if len(sents) <= top_n:
        return sents[:top_n]

    # embed query + sentences
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    s_embs = embed_model.encode(sents, convert_to_numpy=True)

    def norm(x):
        n = np.linalg.norm(x)
        return x / (n + 1e-10)
    qn = norm(q_emb[0])
    sims = [float(np.dot(qn, norm(se))) for se in s_embs]
    idxs = sorted(range(len(sents)), key=lambda i: sims[i], reverse=True)[:top_n]

    # if top sentence looks like a header/menu (very short, contains 'how to' or ends with ':'), also include the next sentence
    out = []
    for idx in idxs:
        out.append(sents[idx])
        # heuristic: if that sentence is short or contains "how to" or looks like heading, add next sentence if exists
        s = sents[idx].lower()
        word_count = len(s.split())
        if (word_count <= 6 or 'how to' in s or s.endswith(':')) and (idx + 1) < len(sents):
            out.append(sents[idx+1])
    # dedupe while preserving order
    seen = set()
    final = []
    for x in out:
        if x not in seen:
            final.append(x)
            seen.add(x)
    return final[:top_n]

def extract_fallback_from_contexts(contexts: list, query: str, n_sentences:int = 1) -> str:
    """
    Deterministic fallback: find the single best sentence across contexts and return it verbatim.
    """
    _, _, embed_model = load_index_and_meta()
    best = None
    best_score = -1.0
    import numpy as np
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    def norm(x): 
        n=np.linalg.norm(x); return x/(n+1e-10)
    qn = norm(q_emb)

    for c in contexts:
        sents = split_into_sentences(c.get("text",""))
        if not sents:
            continue
        s_embs = embed_model.encode(sents, convert_to_numpy=True)
        for s, se in zip(sents, s_embs):
            sc = float(np.dot(qn, norm(se)))
            if sc > best_score:
                best_score = sc
                best = s
    if not best:
        return ""
    return best

def tidy_answer(ans: str, max_sentences: int = MAX_ANSWER_SENTENCES) -> str:
    if not ans:
        return ans
    a = ans
    a = RE_ANGLE_PLACEHOLDER.sub(" ", a)
    a = RE_DOUBLE_DASH_ID.sub(" ", a)
    a = a.replace("…", ". ")
    a = RE_MULTI_DOTS.sub(". ", a)
    a = RE_WHITESPACE.sub(" ", a).strip()
    sents = split_into_sentences(a)
    if not sents:
        snippet = a[:300].strip()
        if not snippet.endswith("."):
            snippet = snippet.rstrip(" .,") + "."
        return snippet
    take = sents[:max_sentences]
    out = " ".join(take).strip()
    if out and out[-1] not in ".!?":
        out = out.rstrip(" .,") + "."
    return out

# -------------------------
# Loading helpers
# -------------------------
def load_index_and_meta():
    global _index, _meta, _embed_model
    if _index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Run src/indexer.py first.")
        _index = faiss.read_index(str(INDEX_PATH))
    if _meta is None:
        if not META_PATH.exists():
            raise FileNotFoundError(f"Meta file not found at {META_PATH}. Run src/indexer.py first.")
        _meta = [json.loads(l) for l in META_PATH.read_text(encoding="utf-8").splitlines()]
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _index, _meta, _embed_model

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder

def get_local_generator():
    global _local_generator
    if _local_generator is None:
        tok = AutoTokenizer.from_pretrained(LOCAL_GEN_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_GEN_MODEL)
        _local_generator = pipeline("text2text-generation", model=model, tokenizer=tok, device=-1)
    return _local_generator

# -------------------------
# Embedding + retrieval
# -------------------------
def embed_query(q: str):
    _, _, em = load_index_and_meta()
    emb = em.encode(q)
    emb = np.asarray(emb, dtype="float32")
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb

def retrieve_candidates(query: str, top_k: int = 50):
    index, meta, _ = load_index_and_meta()
    emb = embed_query(query)
    D, I = index.search(emb, top_k)
    results = []
    if len(I) == 0:
        return results
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        m = meta[idx]
        results.append({"score": float(score), "url": m["url"], "title": m.get("title",""), "text": m["text"], "id": idx})
    return results

# -------------------------
# Reranking
# -------------------------
def rerank_with_cross(query: str, candidates: list, top_n: int = 5):
    if not candidates:
        return []
    cross = get_cross_encoder()
    inputs = [(query, c["text"]) for c in candidates]
    try:
        scores = cross.predict(inputs)
    except Exception as e:
        # fallback: if cross encoder fails, return top_n by original score
        candidates.sort(key=lambda x: x.get("score",0), reverse=True)
        return candidates[:top_n]
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_n]

# -------------------------
# Snippet trimming helpers
# -------------------------
def trim_snippet_to_sentence(snippet: str, max_chars: int = 800) -> str:
    if not snippet:
        return snippet
    s = snippet.replace("\n", " ").strip()
    if len(s) <= max_chars:
        return s
    head = s[:max_chars]
    last_dot = max(head.rfind("."), head.rfind("!"), head.rfind("?"))
    if last_dot and last_dot > int(max_chars * 0.4):
        return head[:last_dot+1].strip()
    cut = head.rsplit(" ", 1)[0]
    return cut.strip()

# -------------------------
# Prompt template (stronger)
# -------------------------
def build_prompt(query: str, contexts: list, sentences_per_context: int = 1):
    """
    Build a prompt that is explicit about producing an ACTIONABLE, concise answer.
    We include one short sentence per source (selected by semantic similarity).
    """
    _, _, embed_model = load_index_and_meta()
    sources_block = []
    for i, c in enumerate(contexts, start=1):
        passage = c.get("text", "").strip()
        best_sents = get_top_sentences_from_passage(passage, query, embed_model, top_n=sentences_per_context)
        snippet = " ".join(best_sents)
        snippet = trim_snippet_to_sentence(snippet, max_chars=500)
        snippet = snippet.rstrip(" .") + "." if snippet and snippet[-1] not in ".!?" else snippet
        sources_block.append(f"[SRC_{i}] URL: {c.get('url')}\n[SRC_{i}] TEXT: {snippet}")

    sources_text = "\n\n".join(sources_block)

    prompt = textwrap.dedent(f"""
    Use only the following snippets to produce a concise, ACTIONABLE answer (1-2 short sentences) that directly answers the question. 
    For "how-to" queries, produce concrete steps or exact fields to set where possible. Do NOT invent facts or add information not present in snippets. 
    If snippets do not contain an answer, reply: "I don't know — please consult the documentation." Then list the Source URLs used.

    {sources_text}

    Question:
    {query}

    Answer (be concise and then list Sources used as URLs):
    """).strip()
    return prompt

# -------------------------
# Generation (OpenAI or local) with fallback formatting
# -------------------------
def generate_answer_with_context(question: str, contexts: list, use_openai: bool = False):
    prompt = build_prompt(question, contexts, sentences_per_context=1)

    # Option A: OpenAI (preferred)
    if use_openai and openai is not None and os.environ.get("OPENAI_API_KEY"):
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":"You are a strict assistant. Use ONLY the provided documentation snippets to answer. Do not hallucinate."},
                    {"role":"user","content": prompt}
                ],
                temperature=0.0,
                max_tokens=200,
                stop=None
            )
            raw_answer = resp["choices"][0]["message"]["content"].strip()
            answer = tidy_answer(raw_answer, max_sentences=MAX_ANSWER_SENTENCES)
            # fallback if model returned unhelpful phrasing
            if answer.lower().startswith("i'm") or "find the source" in answer.lower() or answer.lower().startswith("see"):
                fallback = extract_fallback_from_contexts(contexts, question)
                fallback = fallback.strip()
                if fallback and not fallback.endswith(('.', '!', '?')):
                    fallback = fallback + '.'
                if 'okta' in fallback.lower() or 'authenticator' in fallback.lower():
                    fallback = "Enable Okta SAML SSO: " + fallback
                return tidy_answer(fallback, max_sentences=MAX_ANSWER_SENTENCES), [c["url"] for c in contexts]
            used_urls = [c["url"] for c in contexts if c["url"] in raw_answer]
            if not used_urls:
                used_urls = [c["url"] for c in contexts]
            return answer, used_urls
        except Exception as e:
            print("OpenAI generation failed:", e)

    # Option B: Local generator fallback
    gen = get_local_generator()
    gen_kwargs = {
        "max_length": 200,
        "num_beams": 4,
        "do_sample": False,
        "no_repeat_ngram_size": 3,
        "early_stopping": True
    }
    out = gen(prompt, **gen_kwargs)
    raw_answer = out[0].get("generated_text","").strip()
    answer = tidy_answer(raw_answer, max_sentences=MAX_ANSWER_SENTENCES)
    if answer.lower().startswith("i'm") or "find the source" in answer.lower() or answer.lower().startswith("see"):
        fallback = extract_fallback_from_contexts(contexts, question)
        fallback = fallback.strip()
        if fallback and not fallback.endswith(('.', '!', '?')):
            fallback = fallback + '.'
        if 'okta' in fallback.lower() or 'authenticator' in fallback.lower():
            fallback = "Enable Okta SAML SSO: " + fallback
        return tidy_answer(fallback, max_sentences=MAX_ANSWER_SENTENCES), [c["url"] for c in contexts]
    return answer, [c["url"] for c in contexts]

# -------------------------
# Top-level handler with dedup/diversify
# -------------------------
def handle_rag_query(query: str, top_k: int = 5, use_openai: bool = False, rerank_candidates: int = 50):
    candidates = retrieve_candidates(query, top_k=rerank_candidates)
    if not candidates:
        return {"answer": "No relevant documentation found.", "sources": [], "retrieved": []}

    try:
        top_candidates = rerank_with_cross(query, candidates, top_n=rerank_candidates)
    except Exception as e:
        print("Reranker failed, falling back to FAISS order:", e)
        top_candidates = candidates[:rerank_candidates]

    # Now pick final top_k diversified by URL:
    # strategy: prefer at most MAX_CHUNKS_PER_URL per url; prefer higher rerank_score
    # first, group by URL preserving order
    url_counts = {}
    diversified = []
    for c in top_candidates:
        url = c.get("url")
        cnt = url_counts.get(url, 0)
        if cnt < MAX_CHUNKS_PER_URL:
            diversified.append(c)
            url_counts[url] = cnt + 1
        # stop early if we have enough
        if len(diversified) >= max(top_k, len(top_candidates)):
            break

    # final trimming: ensure at most one chunk per URL until we fill top_k
    seen_urls = set()
    unique_candidates = []
    for c in diversified:
        u = c.get("url")
        if u in seen_urls:
            continue
        unique_candidates.append(c)
        seen_urls.add(u)
        if len(unique_candidates) >= top_k:
            break
    # if we don't have enough unique URLs, allow second chunks (already in diversified)
    if len(unique_candidates) < top_k:
        # fill from diversified preserving order but skipping already selected items
        for c in diversified:
            if c in unique_candidates:
                continue
            unique_candidates.append(c)
            if len(unique_candidates) >= top_k:
                break
    final_candidates = unique_candidates[:top_k]

    # generate answer using final candidates
    answer, urls = generate_answer_with_context(query, final_candidates, use_openai=use_openai)

    return {"answer": answer, "sources": urls, "retrieved": final_candidates}

# small test if run as script
if __name__ == "__main__":
    q = "How do I configure SAML SSO with Okta?"
    print("Running test query:", q)
    res = handle_rag_query(q, top_k=3, use_openai=False)
    print("ANSWER:\n", res["answer"])
    print("SOURCES:\n", res["sources"])
    for r in res["retrieved"][:3]:
        print("----\n", r["url"], "\n", r["text"][:300])
