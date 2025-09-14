import streamlit as st
import pandas as pd
import json
from pathlib import Path
from data_loader import load_tickets
from classifier import classify_ticket, classify_all_and_save


try:
    from rag import handle_rag_query
except Exception:
    handle_rag_query = None  


st.set_page_config(page_title="Atlan - Support Copilot (Phase 3)", layout="wide")
ROOT = Path(__file__).parent.parent.resolve()   
CLASSIFIED_PATH = ROOT.joinpath("classified_tickets_phase2.json")

st.title("Atlan — Support Copilot (Phase 3)")
st.markdown(
    "**Phase 3:** Zero-shot topic classification + HF sentiment + rule-based priority + RAG (retrieval-augmented generation). "
    "This demo shows bulk classification and an interactive agent with RAG."
)


st.sidebar.header("Controls")
use_saved = st.sidebar.checkbox("Load pre-saved classified file (if available)", value=True)
run_classify_all = st.sidebar.button("Classify ALL tickets & Save (Phase 2)")
reload_ui = st.sidebar.button("Reload UI")


st.sidebar.markdown("### RAG options")
use_openai = st.sidebar.checkbox("Use OpenAI for generation (if API key set)", value=False)
top_k = st.sidebar.slider("RAG: number of passages to retrieve", min_value=1, max_value=10, value=5)


if reload_ui:
    try:
        st.experimental_rerun()
    except Exception:
        st.info("Automatic reload isn't supported by this Streamlit version. Please refresh the browser page to reload the UI.")


try:
    tickets = load_tickets()   
except Exception as e:
    st.error("Could not load sample tickets. Ensure sample_tickets.json exists at the project root (one level above src/).")
    st.exception(e)
    tickets = []


if run_classify_all:
    with st.spinner("Running classification on all tickets (models may load on first run)..."):
        try:
            out_path = classify_all_and_save()  
            st.success(f"Classified and saved to: {out_path}")
        except Exception as e:
            st.error("Error during batch classification. See details below.")
            st.exception(e)


classified_data = None
if use_saved and CLASSIFIED_PATH.exists():
    try:
        classified_data = json.loads(CLASSIFIED_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning("Could not read the saved classified file; falling back to live classification.")
        st.exception(e)

tab1, tab2 = st.tabs(["Bulk Classification Dashboard", "Interactive Agent (demo + RAG)"])

with tab1:
    st.header("Bulk ticket classification")
    st.write("This view shows all tickets with their inferred topic tags, sentiment, and priority.")

    rows = []

    if classified_data:
        for entry in classified_data:
            c = entry.get("classification", {})
            rows.append({
                "id": entry.get("id"),
                "subject": entry.get("subject"),
                "topic_tags": ", ".join(c.get("topic_tags", [])),
                "sentiment": c.get("sentiment", ""),
                "priority": c.get("priority", ""),
            })
    else:
        
        with st.spinner("Classifying tickets (zero-shot)... this may take a few seconds on first run"):
            for t in tickets:
                try:
                    c = classify_ticket(t)
                except Exception as e:
                    st.error(f"Error classifying ticket {t.get('id')}: {e}")
                    c = {"topic_tags": [], "sentiment": "Error", "priority": "Error"}
                rows.append({
                    "id": t.get("id"),
                    "subject": t.get("subject"),
                    "topic_tags": ", ".join(c.get("topic_tags", [])),
                    "sentiment": c.get("sentiment", ""),
                    "priority": c.get("priority", ""),
                })

    df = pd.DataFrame(rows)
    
    cols = st.columns([2, 1, 1, 1])
    with cols[0]:
        q = st.text_input("Filter by subject/text contains")
    with cols[1]:
        sel_topic = st.selectbox("Filter by topic (contains)", options=["(any)"] + sorted({t for row in rows for t in row["topic_tags"].split(", ") if t}))
    with cols[2]:
        sel_sent = st.selectbox("Filter by sentiment", options=["(any)","Angry","Frustrated","Neutral","Curious","Positive"])
    with cols[3]:
        sel_prio = st.selectbox("Filter by priority", options=["(any)","P0","P1","P2"])

    df_display = df.copy()
    if q:
        df_display = df_display[df_display["subject"].str.contains(q, case=False, na=False) | df_display["topic_tags"].str.contains(q, case=False, na=False)]
    if sel_topic and sel_topic != "(any)":
        df_display = df_display[df_display["topic_tags"].str.contains(sel_topic, na=False)]
    if sel_sent and sel_sent != "(any)":
        df_display = df_display[df_display["sentiment"] == sel_sent]
    if sel_prio and sel_prio != "(any)":
        df_display = df_display[df_display["priority"] == sel_prio]

    st.dataframe(df_display.reset_index(drop=True), use_container_width=True, height=420)

    st.markdown("### Sample ticket detail")
    
    ids = df_display["id"].tolist()
    if ids:
        sel = st.selectbox("Select ticket", ids)
        
        selected_full = None
        if classified_data:
            selected_full = next((x for x in classified_data if x["id"] == sel), None)
        if not selected_full:
            selected_full = next((x for x in tickets if x["id"] == sel), None)

        st.write(selected_full)
        st.markdown("**Classification (raw)**")
        if selected_full and "classification" in selected_full:
            st.json(selected_full["classification"])
        else:
            
            with st.spinner("Classifying selected ticket..."):
                try:
                    c = classify_ticket(selected_full)
                except Exception as e:
                    st.error("Error during classification of selected ticket.")
                    st.exception(e)
                    c = {}
                st.json(c)
    else:
        st.info("No tickets to display with current filters.")

with tab2:
    st.header("Interactive Agent (Phase 3 - analysis + RAG)")
    st.markdown(
        "Paste a ticket subject and body (or type). The backend analysis will show topic tags, sentiment and priority. "
        "If the topic is one of the RAG-enabled categories (How-to, Product, Best practices, API/SDK, SSO), the app will run RAG and show a cited answer."
    )

    user_input = st.text_area("Paste a ticket subject + body (or type a new one)", height=220, placeholder="Subject line on first line, body below...")
    analyze = st.button("Analyze input")

    if analyze:
        if not user_input.strip():
            st.warning("Enter some ticket text to analyze.")
        else:
            
            lines = user_input.strip().split("\n")
            subject = lines[0]
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else user_input.strip()
            demo_ticket = {"id": "TEMP", "subject": subject, "body": body}
            with st.spinner("Analyzing (zero-shot + sentiment)..."):
                try:
                    c = classify_ticket(demo_ticket)
                except Exception as e:
                    st.error("Error during classification.")
                    st.exception(e)
                    c = {"topic_tags": [], "sentiment": "Error", "priority": "Error"}

            st.subheader("Internal analysis (backend view)")
            st.json(c)

            st.subheader("Final response (frontend view)")
            
            allowed_rag = {"How-to", "Product", "Best practices", "API/SDK", "SSO"}

            
            if any(lbl in allowed_rag for lbl in c.get("topic_tags", [])):
                if handle_rag_query is None:
                    st.error("RAG handler not found. Make sure src/rag.py exists and is importable.")
                else:
                    st.info("RAG triggered — retrieving docs and generating an answer...")
                    with st.spinner("Retrieving + generating answer (may take a few seconds)..."):
                        
                        query_text = f"{subject}\n\n{body}"
                        try:
                            rag_res = handle_rag_query(query_text, top_k=top_k, use_openai=use_openai)
                        except Exception as e:
                            st.error("Error during RAG operation.")
                            st.exception(e)
                            rag_res = {"answer": "RAG failed.", "sources": [], "retrieved": []}

                    st.subheader("Answer")
                    st.markdown(rag_res.get("answer", "No answer returned."))

                    st.subheader("Sources (citations)")
                    for s in rag_res.get("sources", []):
                        st.write(s)

                    st.subheader("Top retrieved passages (debug view)")
                    for r in rag_res.get("retrieved", [])[:top_k]:
                        st.markdown(f"**Title:** {r.get('title','(no title)')}  \n**URL:** {r.get('url')}  \n**Score:** {r.get('score'):.4f}")
                        st.write(r.get("text","")[:800] + ("..." if len(r.get("text","")) > 800 else ""))

            else:
                st.success(f"This ticket has been classified as {c.get('topic_tags', [])} and routed to the appropriate team.")

st.markdown("---")
st.caption(
    "Phase 3 demo — zero-shot topic classification (facebook/bart-large-mnli), sentiment (distilbert SST-2), and RAG using local FAISS + sentence-transformers. "
    "Toggle 'Use OpenAI' in the sidebar to use the OpenAI API for generation (requires OPENAI_API_KEY in env)."
)
