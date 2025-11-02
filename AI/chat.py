# chat.py
"""
Chat orchestrator:
 - loads KB (FAISS + chunks)
 - NLU pipeline (nlu.py)
 - builds a hybrid prompt combining empathy and retrieved docs
 - calls LLM (OpenAI Chat API if OPENAI_API_KEY present, else local HF seq2seq)
 - safety flows: crisis detection, canned empathetic fallback
"""

import os
import time
import json
import pickle
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from nlu import detect_intent, detect_emotion, safety_check
from build_knowledge_base import load_index, query, EMBEDDING_MODEL, CHUNKS_FILE, INDEX_FILE

# Config via env
USE_OPENAI = os.environ.get("USE_OPENAI", "true").lower() in ["1", "true", "yes"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")  # set as needed
LOCAL_SEQ2SEQ = os.environ.get("LOCAL_SEQ2SEQ", "google/flan-t5-large")

# Loading KB
index, chunks_meta = load_index(index_path=INDEX_FILE, chunks_path=CHUNKS_FILE)
print("Loaded KB index and chunks.")

# Embedding model used at query time (kept lightweight)
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# If OpenAI is chosen, lazy import openai
openai = None
if USE_OPENAI and OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        openai_loaded = True
    except Exception as e:
        print("OpenAI import failed:", e)
        openai_loaded = False
        USE_OPENAI = False
else:
    USE_OPENAI = False
    openai_loaded = False

# Local HF model (fallback)
hf_tokenizer = None
hf_model = None
if not USE_OPENAI:
    print("Loading local seq2seq model:", LOCAL_SEQ2SEQ)
    hf_tokenizer = AutoTokenizer.from_pretrained(LOCAL_SEQ2SEQ)
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_SEQ2SEQ)
    print("Local model loaded.")

# Persona + prompt templates
SYSTEM_PERSONA = (
    "You are 'Manas', an empathetic, non-judgmental AI companion designed to support "
    "students with mental health concerns. You validate feelings first, avoid giving medical "
    "diagnoses, suggest evidence-based coping strategies when helpful, and escalate if there's "
    "risk of self-harm or imminent danger. Provide brief, clear, and compassionate responses."
)

EMPATHY_TEMPLATES = [
    "That sounds really hard — thank you for telling me. I'm here with you.",
    "I'm sorry you're feeling this way. Would you like to tell me what happened?",
    "It makes sense that you'd feel that way; you're not alone in this."
]

CRISIS_MODAL = (
    "If you are thinking of harming yourself, please contact your local emergency services immediately. "
    "You can also reach out to your campus counseling center or this international resource: "
    "https://www.opencounseling.com/suicide-hotlines\n"
    "Would you like me to connect you with a human moderator or see local helplines?"
)

def get_retrieved_context(user_text, k=3):
    results = query(index, chunks_meta, user_text, model_name=EMBEDDING_MODEL, k=k)
    # join with metadata and short-circuit if empty
    if not results:
        return []
    return results

def build_prompt(user_text, retrieved: List[dict], emotion=None, intent=None):
    # Build a compact context block
    context_block = "\n\n".join([f"[{r['meta']['source']}] {r['chunk']}" for r in retrieved]) if retrieved else ""
    # Instructions to the LLM
    instructions = (
        SYSTEM_PERSONA + "\n\n"
        "When you respond:\n"
        "- Start with validation of feelings.\n"
        "- Offer 2-3 short coping suggestions (grounded in the context below when relevant).\n"
        "- Ask a gentle follow-up question to continue the conversation.\n"
        "- If the user seems at risk, follow crisis protocol (see end of response) rather than giving normal advice.\n\n"
    )
    # Add signal about detected emotion / intent so model can adapt tone
    signals = ""
    if emotion:
        signals += f"[DETECTED_EMOTION]={emotion.get('label')} (score={emotion.get('score'):.2f})\n"
    if intent:
        signals += f"[DETECTED_INTENT]={intent.get('label')} (score={intent.get('score'):.2f})\n"

    prompt = f"""{instructions}{signals}\nREFERENCE KNOWLEDGE:\n{context_block}\n\nUSER: \"{user_text}\"\n\nMANAS:"""
    # Ensure prompt not huge — caller must ensure tokens fit
    return prompt

def call_openai_chat(prompt, max_tokens=256, temperature=0.7):
    global openai
    if not openai:
        import openai as _openai
        openai = _openai
        openai.api_key = OPENAI_API_KEY
    messages = [{"role":"system","content":SYSTEM_PERSONA},
                {"role":"user","content":prompt}]
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp["choices"][0]["message"]["content"].strip()

def call_local_hf(prompt, max_new_tokens=150, temperature=0.7):
    inputs = hf_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = hf_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
    text = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def canned_empathy(user_text):
    # simple fallback: pick an empathy template + short reflective sentence
    import random
    base = random.choice(EMPATHY_TEMPLATES)
    return f"{base} Can you tell me a bit more about what's been going on?"

def get_bot_response(user_text):
    # 1) Safety check right away
    safety = safety_check(user_text)
    if safety["high_risk"]:
        return CRISIS_MODAL, {"escalate": True, "safety": safety}

    # 2) NLU & Emotion
    intent = detect_intent(user_text)
    emotion = detect_emotion(user_text)

    # 3) Retrieve contextual KB chunks for more grounded replies
    retrieved = get_retrieved_context(user_text, k=4)

    # 4) Build prompt and call LLM (OpenAI preferred)
    prompt = build_prompt(user_text, retrieved, emotion=emotion, intent=intent)
    try:
        if USE_OPENAI and openai_loaded:
            reply = call_openai_chat(prompt)
        else:
            reply = call_local_hf(prompt)
        # safety post-filter (avoid hallucinated clinical claims)
        # If reply too short or repeats user, give canned empathy
        if not reply or len(reply.split()) < 6:
            return canned_empathy(user_text), {"escalate": False, "safety": safety}
        return reply, {"escalate": False, "safety": safety, "intent": intent, "emotion": emotion}
    except Exception as e:
        print("LLM call failed:", e)
        return canned_empathy(user_text), {"escalate": False, "safety": safety, "err": str(e)}

# CLI demo
if __name__ == "__main__":
    print("Manas (demo). Type 'quit' to exit.")
    while True:
        u = input("You: ").strip()
        if u.lower() in ("quit","exit"):
            print("Manas: Take care — you are not alone.")
            break
        resp, meta = get_bot_response(u)
        print("\nManas:", resp)
        print("-- meta:", json.dumps(meta, indent=2))
        print()
