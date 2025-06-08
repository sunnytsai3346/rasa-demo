from datetime import datetime
import json
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import fitz

from actions.logger_util import log_debug  # PyMuPDF


#try different hugging face SentenceTransformer
#1. all-MiniLM-L12-v2
#Better accuracy than L6-v2, but slightly slower.
#Size: ~80MB
#Great for general-purpose semantic similarity.

#2. all-mpnet-base-v2
#One of the most accurate general-purpose English models in SentenceTransformers.
#Larger and slower, but significantly better embeddings.
#Good for semantic search and clustering.

CACHE_PATH = os.path.join(os.path.dirname(__file__), "cached_sections.json")        
class PDFKnowledgeBase:
    def __init__(self, pdf_path,debug=False):
        self.debug = debug        
        
        # SentenceTransformer model before extract_sections, so I can debug 
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        if os.path.exists("cached_sections.json"):
            if self.debug:
                print("[DEBUG] Loading sections from cache.")
                with open("cached_sections.json", encoding="utf-8") as f:
                    self.sections = json.load(f)
        else:
            self.sections = self.extract_sections(pdf_path)  
        if self.sections is None:
            raise ValueError("extract_sections() returned None instead of a list")
       
        self.embeddings = self.embed_sections()

    def load_pdf_sections(self, path):
        doc = fitz.open(path)
        chunks = []
        for page in doc:
            text = page.get_text()
            if len(text) > 500:
                chunks.append(text)
        return chunks

    def extract_sections(self, path):
        doc = fitz.open(path)
        section_data = []

        for page in doc:
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: b[1])  # top to bottom

            for block in blocks:
                text = block[4].strip()
                if len(text.split()) < 10 and text.istitle():  # naive title check
                    title = text
                    continue
                if text and title:
                    section_data.append({"title": title, "content": text})
                    log_debug(title, text, self.summarize(text))
                    if self.debug:
                        print(f"\n[EXTRACTED SECTION]")
                        print(f"[{datetime.now().isoformat(timespec='seconds')}]")
                        print(f"Title : {title}")
                        print(f"Content:\n{text}")  #
                        print(f"summary:\n{self.summarize(text)}")  # 
                    title = None

        # Summarize
        for section in section_data:
            section["summary"] = self.summarize(section["content"])
        # After parsing:
        if self.debug:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(section_data, f, ensure_ascii=False, indent=2)
        
        return section_data   

    def summarize(self, text):
        if len(text) < 400:
            return text.strip()
        summary = self.summarizer(text[:1024], max_length=200, min_length=60, do_sample=False)
        return summary[0]["summary_text"]

    def embed_sections(self):
        return self.embedder.encode(self.sections, convert_to_tensor=True)

    def embed_titles(self):
        titles = [section["title"] for section in self.sections]
        return self.embedder.encode(titles, convert_to_tensor=True)    

    def search(self, query, top_k=1):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)[0]
        best_match = self.sections[hits[0]["corpus_id"]]
        print('search,',best_match)
        return self.summarize(best_match)
    def search_by_title(self, query, top_k=1):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.section_embeddings, top_k=1)
        if not hits or not hits[0]:
            return "Not Found", "Sorry, I couldn’t find a matching section."
        
        best_hit = hits[0][0]
        section_index = best_hit["corpus_id"]
        match = self.sections[section_index]
        return match["title"], match["summary"]

