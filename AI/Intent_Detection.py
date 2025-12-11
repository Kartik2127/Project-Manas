
from transformers import pipeline
import os
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

emotion_clf = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

safety_zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


INTENT_CANDIDATES = [
    "greeting", "goodbye", "smalltalk", "ask-resource", "feel_anxious",
    "feel_depressed", "seek_coping", "ask_for_professional_help", "self_harm_ideation",
    "gratitude", "neutral"
]

SAFETY_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die", "hurt myself",
    "cut myself", "hang myself", "overdose", "cant go on"
]

def detect_intent(text, candidates=INTENT_CANDIDATES):
    """
    Returns: {label: str, score: float, all_scores: list}
    """
    out = zero_shot(text, candidate_labels=candidates, multi_label=False)
    return {"label": out["labels"][0], "score": out["scores"][0], "all": list(zip(out["labels"], out["scores"]))}

def detect_emotion(text):
    """
    Returns primary emotion label, e.g. 'sadness', 'joy', 'anger', ...
    """
    out = emotion_clf(text)[0]
    return {"label": out["label"], "score": out["score"]}

def simple_keyword_safety(text):
    t = text.lower()
    for kw in SAFETY_KEYWORDS:
        if kw in t:
            return True, kw
    return False, None

def safety_check(text):
    has_kw, kw = simple_keyword_safety(text)
    if has_kw:
        return {"high_risk": True, "reason": f"keyword:{kw}", "confidence": 0.99}

    candidate_labels = ["self-harm or suicidal", "not suicidal"]
    out = safety_zero_shot(text, candidate_labels)
    if out["labels"][0] == "self-harm or suicidal" and out["scores"][0] > 0.7:
        return {"high_risk": True, "reason": "zero_shot_selfharm", "confidence": float(out["scores"][0])}
    return {"high_risk": False, "reason": "none_detected", "confidence": float(out["scores"][0])}
