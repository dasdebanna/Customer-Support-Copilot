# Dockerfile — explicit copy of runtime files for Hugging Face Spaces
FROM python:3.13.5-slim

# avoid interactive prompts during apt operations
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# install minimal build deps some Python packages may need
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy the launcher, source code and config into the image
COPY streamlit_app.py /app/streamlit_app.py
COPY src/ /app/src/

# copy runtime/sample/index/meta files explicitly into /app
# (these are referenced by your app at runtime)
COPY sample_tickets.json /app/sample_tickets.json
COPY docs_corpus.jsonl /app/docs_corpus.jsonl
COPY docs_meta.jsonl /app/docs_meta.jsonl
# optional: FAISS binary (if you want it baked into the image)
# If it's tracked via Git LFS this will still copy the pointer; LFS objects need to be handled appropriately.
COPY faiss_index.bin /app/faiss_index.bin

# copy streamlit config (if present)
COPY .streamlit /app/.streamlit

# expose default port (HF overrides with its own environment)
EXPOSE 8501

# healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# use the launcher in the repo root (streamlit_app.py)
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.headless=true"]
