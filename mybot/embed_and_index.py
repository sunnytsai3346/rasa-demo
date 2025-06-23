# Install requirements first:
# pip install sentence-transformers faiss-cpu PyMuPDF ollama
# One-time PDF load and FAISS save

import json
import os
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import uuid

PDF_DIR = os.path.join(os.path.dirname(__file__), "Actions","DATA","PDF")
EMBED_MODEL = "intfloat/e5-large-v2"
INDEX_PATH = os.path.join(os.path.dirname(__file__), "Actions","DATA","FAISS_index")
CHUNK_SIZE = 300

# 1. Load and chunk PDFs
def load_pdfs_to_chunks(pdf_dir):
    chunks = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            doc = fitz.open(os.path.join(pdf_dir, file))
            for page in doc:
                text = page.get_text()
                if len(text.strip()) > 100:
                    chunks.append({"text": text,
                                   "source": file,  # flattened for easy access later,
                                   "meta": {"file": file}})
    return chunks

# 2. Embed chunks
def embed_chunks(chunks, model):
    texts = [f"query: {chunk['text']}" for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

# 3. Save to Faiss

def save_faiss_index(chunks, embeddings):
    print('save_faiss_index')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(INDEX_PATH, "index.faiss"))

    # Save metadata
    with open(os.path.join(INDEX_PATH, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)
        


# ---- To prepare the DB (run once):
if __name__ == "__main__":
    print("Loading and embedding PDFs...")
    model = SentenceTransformer(EMBED_MODEL)
    chunks = load_pdfs_to_chunks(PDF_DIR)
    embeddings = embed_chunks(chunks, model)
    save_faiss_index(chunks, embeddings)
    print("FAISS index saved.")
