
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.tokenize import sent_tokenize
import logging
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

nltk.download('punkt', quiet=True)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
INDEX_FILE = os.environ.get("KB_INDEX_PATH", "knowledge_base.index")
CHUNKS_FILE = os.environ.get("KB_CHUNKS_PATH", "kb_chunks.pkl")
D = None  

def chunk_text(text, max_words=200, overlap_words=30):
    sents = sent_tokenize(text)
    chunks = []
    current = []
    current_len = 0
    for sent in sents:
        words = sent.split()
        if current_len + len(words) <= max_words:
            current.append(sent)
            current_len += len(words)
        else:
            chunks.append(" ".join(current).strip())
            overlap = " ".join(current[-overlap_words:]) if overlap_words and len(current) > overlap_words else ""
            current = [overlap, sent] if overlap else [sent]
            current_len = len((" " + overlap).split()) + len(words) if overlap else len(words)
    if current:
        chunks.append(" ".join(current).strip())
    chunks = [c for c in chunks if len(c.split()) > 8]
    logging.info(f"Produced {len(chunks)} chunks.")
    return chunks

def scrape_article(url):
    logging.info(f"Scraping {url} ...")
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
        else:
            paragraphs = soup.find_all("p")
        text = " ".join([p.get_text(" ", strip=True) for p in paragraphs])
        logging.info(f"Scraped {len(text)} characters.")
        return text
    except Exception as e:
        logging.warning(f"Failed to scrape: {e}")
        return ""

def embed_chunks(chunks, model_name=EMBEDDING_MODEL):
    logging.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    logging.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings, model

def create_faiss_index(embeddings, index_path=INDEX_FILE):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, index_path)
    logging.info(f"Saved FAISS index to {index_path}")
    return index

def save_chunks(chunks, chunks_path=CHUNKS_FILE):
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    logging.info(f"Saved chunk metadata to {chunks_path}")

def build_from_texts(texts, model_name=EMBEDDING_MODEL, index_path=INDEX_FILE, chunks_path=CHUNKS_FILE):
    all_chunks = []
    metadata = []
    for doc in texts:
        source = doc.get("source", "unknown")
        src_tags = doc.get("tags", [])
        doc_text = doc.get("text", "")
        chunks = chunk_text(doc_text)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            metadata.append({"source": source, "tags": src_tags, "chunk_id": f"{source}::{i}"})
    embeddings, model = embed_chunks(all_chunks, model_name)
    faiss.normalize_L2(embeddings)
    index = create_faiss_index(embeddings, index_path)
    save_chunks(list(zip(all_chunks, metadata)), chunks_path)
    return index, all_chunks, metadata

def load_index(index_path=INDEX_FILE, chunks_path=CHUNKS_FILE):
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError("Index or chunks file not found. Run build step first.")
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks_meta = pickle.load(f)
    return index, chunks_meta

def query(index, chunks_meta, query_text, model_name=EMBEDDING_MODEL, k=4):
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb.astype('float32'), k)
    results = []
    for idx in indices[0]:
        if idx < len(chunks_meta):
            chunk, meta = chunks_meta[idx]
            results.append({"chunk": chunk, "meta": meta})
    return results

if __name__ == "__main__":
    urls = [
        "https://www.who.int/news-room/fact-sheets/detail/depression"
    ]
    docs = []
    for i, u in enumerate(urls):
        txt = scrape_article(u)
        if txt:
            docs.append({"source": f"who_depression_{i}", "text": txt, "tags": ["who", "depression"]})
    if docs:
        build_from_texts(docs)
        print("KB built successfully.")
    else:
        print("No documents to build from.")
