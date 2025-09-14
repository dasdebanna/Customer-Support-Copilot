from typing import Dict, List, Union
from transformers import pipeline
import math
import json
from pathlib import Path


_zero_shot_clf = None
_sentiment_clf = None

def get_zero_shot_classifier():
    global _zero_shot_clf
    if _zero_shot_clf is None:
        
        _zero_shot_clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _zero_shot_clf

def get_sentiment_classifier():
    global _sentiment_clf
    if _sentiment_clf is None:
        
        _sentiment_clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return _sentiment_clf


TOPIC_LABELS = [
    "How-to",
    "Product",
    "Connector",
    "Lineage",
    "API/SDK",
    "SSO",
    "Glossary",
    "Best practices",
    "Sensitive data"
]


LABEL_DESCRIPTIONS = {
    "How-to": "user asking how to perform a task or request a tutorial",
    "Product": "product feature, UI or general product question",
    "Connector": "questions about connectors, crawlers, integrations and failures",
    "Lineage": "questions about lineage, upstream/downstream or lineage exports",
    "API/SDK": "developer questions about APIs, SDKs, endpoints, code examples",
    "SSO": "authentication, SAML, SSO, Okta, login issues",
    "Glossary": "business glossary, terms, bulk import of glossary terms",
    "Best practices": "request for recommended approach, best practices or governance",
    "Sensitive data": "questions about PII, masking, DLP, secrets"
}

def classify_topic_zero_shot(text: str, labels: List[str] = TOPIC_LABELS, hypothesis_template: str = "This text is about {}.") -> Dict:
    """
    Returns a dictionary with labels and scores from zero-shot classifier.
    """
    clf = get_zero_shot_classifier()
    
    res = clf(sequences=text, candidate_labels=labels, hypothesis_template=hypothesis_template)

    return res

def classify_sentiment_hf(text: str) -> str:
    """
    Returns a human-friendly sentiment label, mapping HF outputs to your schema.
    HF model returns POSITIVE/NEGATIVE with a score.
    We'll use a small mapping to Frustrated/Curious/Angry/Neutral/Positive.
    """
    clf = get_sentiment_classifier()
    out = clf(text[:1000])  
    
    if not out:
        return "Neutral"
    lab = out[0]["label"].upper()
    score = out[0]["score"]
    
    if lab == "NEGATIVE":
        
        if score > 0.9:
            return "Angry"
        return "Frustrated"
    elif lab == "POSITIVE":
        if score > 0.9:
            return "Positive"
        return "Curious"
    else:
        return "Neutral"


PRIORITY_KEYWORDS_P0 = ["urgent", "asap", "blocked", "blocker", "critical", "production", "failed", "failure", "infuriating", "can't", "cant", "down", "urgent:"]
PRIORITY_KEYWORDS_P1 = ["need", "important", "deadline", "next week", "approaching", "required", "soon", "high"]

def classify_priority(text: str, subject: str = "") -> str:
    t = (subject + " " + text).lower()
    for k in PRIORITY_KEYWORDS_P0:
        if k in t:
            return "P0"
    for k in PRIORITY_KEYWORDS_P1:
        if k in t:
            return "P1"
    return "P2"

def classify_ticket(ticket: Dict, top_k: int = 2, label_score_threshold: float = 0.25) -> Dict:
    """
    Full classification of a single ticket:
     - topic_tags: top_k labels from zero-shot (above threshold)
     - sentiment: HF sentiment mapped
     - priority: rule-based
    """
    text = " ".join([ticket.get("subject", ""), ticket.get("body", "")])
    z = classify_topic_zero_shot(text)
    labels = z.get("labels", [])
    scores = z.get("scores", [])
    
    topic_tags = []
    for lbl, score in zip(labels, scores):
        if score >= label_score_threshold:
            topic_tags.append(lbl)
        if len(topic_tags) >= top_k:
            break
    
    if not topic_tags and labels:
        topic_tags = [labels[0]]

    sentiment = classify_sentiment_hf(text)
    priority = classify_priority(ticket.get("body",""), ticket.get("subject",""))

    return {
        "id": ticket.get("id"),
        "topic_tags": topic_tags,
        "topic_scores": {lbl: float(s) for lbl, s in zip(labels, scores)},
        "sentiment": sentiment,
        "priority": priority
    }


def classify_all_and_save(input_path: Union[str, Path] = "../sample_tickets.json", output_path: Union[str, Path] = "../classified_tickets_phase2.json"):
    p_in = Path(__file__).parent.joinpath(input_path).resolve()
    p_out = Path(__file__).parent.joinpath(output_path).resolve()
    tickets = json.loads(p_in.read_text(encoding="utf-8"))
    results = []
    for t in tickets:
        c = classify_ticket(t)
        results.append({**t, "classification": c})
    p_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} classified tickets to {p_out}")
    return p_out

if __name__ == "__main__":
    classify_all_and_save()
