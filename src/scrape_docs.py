# src/scrape_docs.py
"""
Crawl allowed Atlan docs and write a cleaned docs_corpus.jsonl.
Improvements:
 - robust cleaning of encoding artifacts (utf-8 replace + ftfy optional)
 - removes paragraph markers ¶, <placeholders>, group-id--digits tokens
 - strips boilerplate lines and tiny nav lines
 - collapses and normalizes whitespace / encoding
 - removes script/style/header/footer/nav/form tags before extracting
Output: docs_corpus.jsonl (overwrites)
"""
import requests
import html
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from pathlib import Path
from url_normalize import url_normalize
import ujson as json
from tqdm import tqdm

OUTPUT = Path(__file__).parent.parent.joinpath("docs_corpus.jsonl")
SEEDS = [
    "https://docs.atlan.com/",
    "https://developer.atlan.com/"
]
ALLOWED_DOMAINS = {"docs.atlan.com", "developer.atlan.com"}
HEADERS = {"User-Agent": "atlan-rag-bot/0.1 (+your_email@example.com)"}

# heuristics
MIN_LINE_WORDS = 3
MIN_PAGE_WORDS = 30

# regex cleanup
RE_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
RE_PARAGRAPH_MARK = re.compile(r"¶")
RE_ANGLE_PLACEHOLDER = re.compile(r"<[^>\n]{1,200}>")
RE_DOUBLE_DASH_ID = re.compile(r"\b[a-zA-Z0-9_-]{3,}--\d{3,}\b")
RE_MULTIPLE_SPACES = re.compile(r"\s+")
RE_REPEATED_CHAR = re.compile(r"(.)\1{5,}")   # long repeated chars
RE_BAD_ELLIPSIS = re.compile(r"\.{2,}")       # multiple dots

BOILERPLATE_KEYWORDS = [
    "table of contents", "overview", "read more", "privacy", "terms", "©", "cookie",
    "search", "related articles", "last updated", "release notes", "subscribe", "breadcrumb"
]

# optional: try to import ftfy for robust fixes (if installed)
try:
    import ftfy
except Exception:
    ftfy = None


def is_allowed(url):
    try:
        return urlparse(url).netloc in ALLOWED_DOMAINS
    except:
        return False

def _keep_line(line: str) -> bool:
    s = line.strip().lower()
    if not s:
        return False
    if len(s.split()) < MIN_LINE_WORDS:
        return False
    if s.startswith("http") or s.startswith("www."):
        return False
    for k in BOILERPLATE_KEYWORDS:
        if k in s:
            return False
    # short code-like lines
    if len(s) < 10 and any(ch in s for ch in ['/', '.', '#']):
        return False
    return True

def clean_text(soup):
    # remove undesired blocks
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
        tag.decompose()
    parts = []
    # only consider headings, paragraphs and list items
    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        t = el.get_text(separator=" ", strip=True)
        if not t:
            continue
        # HTML unescape
        t = html.unescape(t)
        # remove paragraph mark and placeholders
        t = RE_PARAGRAPH_MARK.sub(" ", t)
        t = RE_ANGLE_PLACEHOLDER.sub(" ", t)
        t = RE_DOUBLE_DASH_ID.sub(" ", t)
        # remove control chars
        t = RE_CONTROL.sub(" ", t)
        # remove excessive repeated chars
        t = RE_REPEATED_CHAR.sub(" ", t)
        # normalize ellipsis
        t = RE_BAD_ELLIPSIS.sub(". ", t)
        # collapse whitespace
        t = RE_MULTIPLE_SPACES.sub(" ", t).strip()
        if _keep_line(t):
            parts.append(t)
    joined = "\n\n".join(parts).strip()
    # final normalization: force utf-8 safe output & fix broken chars
    joined = joined.encode('utf-8', errors='replace').decode('utf-8')
    joined = joined.replace("\ufffd", " ")
    # optional stronger fix using ftfy if available
    if ftfy is not None:
        joined = ftfy.fix_text(joined)
    # Remove common weird bytes sequences left by encoding (Â, â etc.)
    joined = joined.replace("Â", "").replace("â", "")
    joined = RE_MULTIPLE_SPACES.sub(" ", joined).strip()
    return joined

def crawl(seeds=SEEDS, max_pages=1000, max_depth=2):
    seen = set()
    out = []
    q = deque()
    for s in seeds:
        q.append((s, 0))
    pbar = tqdm(total=max_pages, desc="Crawl", unit="page")
    while q and len(out) < max_pages:
        url, depth = q.popleft()
        url = url_normalize(url)
        if url in seen:
            continue
        if depth > max_depth:
            continue
        if not is_allowed(url):
            seen.add(url)
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code != 200:
                seen.add(url)
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            title = soup.title.string.strip() if soup.title else url
            text = clean_text(soup)
            if text and len(text.split()) >= MIN_PAGE_WORDS:
                out.append({"url": url, "title": title, "text": text})
                pbar.update(1)
            seen.add(url)
            # find links
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                href = url_normalize(href)
                if is_allowed(href) and href not in seen:
                    # skip common media files
                    if any(href.lower().endswith(ext) for ext in [".pdf", ".zip", ".png", ".jpg", ".jpeg", ".svg"]):
                        continue
                    q.append((href, depth + 1))
        except Exception as e:
            # keep going
            seen.add(url)
            continue
    pbar.close()
    # write JSONL (overwrite)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for doc in out:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} docs to {OUTPUT}")

if __name__ == "__main__":
    crawl(max_pages=400, max_depth=2)
