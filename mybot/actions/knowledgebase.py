
import json
import os
import fitz  # PyMuPDF
from actions.logger_util import log_debug, log_summary_query
import pdfplumber
from collections import defaultdict
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

from actions.logger_util import log_debug, log_debug_all

#try different hugging face SentenceTransformer
#1. all-MiniLM-L12-v2
#Better accuracy than L6-v2, but slightly slower.
#Size: ~80MB
#Great for general-purpose semantic similarity.

#2. all-mpnet-base-v2
#One of the most accurate general-purpose English models in SentenceTransformers.
#Larger and slower, but significantly better embeddings.
#Good for semantic search and clustering.


CACHE_PATH = os.path.join(os.path.dirname(__file__), "debug_summary_log.json")   

class PDFKnowledgeBase:
    def __init__(self, pdf_path,debug=False):
        self.debug = debug        
        
        # SentenceTransformer model before extract_sections, so I can debug 
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        file_exists = os.path.exists(CACHE_PATH)
        if file_exists:
            if self.debug:
                print("[DEBUG-1] Loading sections from cache.")
                with open(CACHE_PATH, encoding="utf-8") as f:
                    self.sections = json.load(f)
        else:
            print("[DEBUG-2] extract sections generate new json.")
            self.sections = self.extract_sections(pdf_path)  
        if self.sections is None:
            raise ValueError("extract_sections() returned None instead of a list")
        
        # Semantic search embeddings
        self.section_titles = [s["title"] for s in self.sections or []]
        
        self.section_summaries = [s["summary"] for s in self.sections]
        #retrieve full original content, not just the summary.
        self.contents = [s["content"] for s in self.sections]
        
        
        
        #Embeddings
        # Title-based search
        self.section_embeddings = self.model.encode(self.contents, convert_to_tensor=True)
        self.title_embeddings = self.model.encode(self.section_titles, convert_to_tensor=True)
        # Full-text (summary) based search
        self.embeddings = self.model.encode(self.section_summaries, convert_to_tensor=True)
        

       
        
        
       

    def load_pdf_sections(self, path):
        doc = fitz.open(path)
        chunks = []
        for page in doc:
            text = page.get_text()
            if len(text) > 500:
                chunks.append(text)
        return chunks
    
    #enhance with font-size
    def extract_sections(self, pdf_path: str) :
        

        self.sections = []
        self.section_embeddings = []
        self.section_titles = []
        self.all_titles = []

        current_section = None
        current_subsection = None
        buffer = []

        def flush_section(title_path, buffer_text):
            if buffer_text.strip():
                full_title = " > ".join(title_path)
                summary = self.summarize(buffer_text)
                embedding = self.model.encode(summary, convert_to_tensor=True)
                self.sections.append({
                    "title": full_title,
                    "summary": summary,
                    "content": buffer_text.strip()
                })
                self.section_titles.append(full_title)
                self.all_titles.append(full_title)
                self.section_embeddings.append(embedding)
                log_debug_all(title_path, buffer_text, summary)
                #if self.debug:
                    #print(f"\n[EXTRACTED SECTION]")
                    #print(f"Title Path: {' > '.join(title_path)}")
                    #print(f"Content:\n{buffer_text[:500]}...")  # Only print first 500 chars
                    #print(f"Summary: {summary[:300]}...\n")

        with pdfplumber.open(pdf_path) as pdf:
            title_path = []
            for page in pdf.pages:
                lines = page.extract_text().split('\n') if page.extract_text() else []

                for line in lines:
                    clean_line = line.strip()

                    # Detect main headers (heuristic: all uppercase and >3 words)
                    if clean_line.isupper() and len(clean_line.split()) >= 3:
                        if buffer:
                            flush_section(title_path, "\n".join(buffer))
                            buffer = []
                        title_path = [clean_line.title()]  # Reset path to new main section
                        continue

                    # Detect subheaders (heuristic: Capitalized sentence, no ending punctuation, and short length)
                    if clean_line and clean_line[0].isupper() and not clean_line.endswith(('.', ':')) and len(clean_line.split()) <= 8:
                        if buffer:
                            flush_section(title_path, "\n".join(buffer))
                            buffer = []
                        if len(title_path) == 0:
                            title_path = ["Untitled"]
                        title_path = title_path[:1] + [clean_line.strip()]  # append subheader
                        continue

                    # Detect table rows: if line has multiple aligned sections or known table headers
                    if len(clean_line.split()) >= 3 and (
                        "Available actions" in clean_line or "Description" in clean_line
                    ):
                        buffer.append("\n--- Table Start ---\n")
                        buffer.append(clean_line)
                        continue

                    if clean_line:
                        buffer.append(clean_line)

            # Final flush at end of document
            if buffer:
                flush_section(title_path, "\n".join(buffer))
        
        print("[DEBUG] Finished extract_sections")
        print(f"[DEBUG] Extracted {len(self.sections)} sections")
        # After parsing:
        if self.debug:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.sections, f, ensure_ascii=False, indent=2)
        
        return self.sections
    
    def looks_like_table(self, text):
        lines = text.split("\n")
        if len(lines) > 5:
            return True
        numeric_lines = sum(1 for line in lines if sum(c.isdigit() for c in line) > len(line) * 0.4)
        return numeric_lines / max(len(lines), 1) > 0.5

    def summarize(self, text):
        max_chunk_words=400
        if len(text) < 400:
            return text.strip()

        try:
            words = text.split()
            chunks = [words[i:i + max_chunk_words] for i in range(0, len(words), max_chunk_words)]

            summaries = []
            for chunk_words in chunks:
                chunk_text = " ".join(chunk_words)
                summary = self.summarizer(
                    chunk_text,
                    max_length=200,
                    min_length=60,
                    do_sample=False
                )
                if summary and "summary_text" in summary[0]:
                    summaries.append(summary[0]["summary_text"])
                else:
                    print("Empty summary for chunk.")
                    summaries.append(chunk_text)

            return " ".join(summaries).strip()

        except Exception as e:
            print("Summarization error:", e)
            return text.strip()

    def search(self, query: str, top_k: int = 3, threshold: float = 0.6, overview_mode: bool = True):
        
        
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, self.section_embeddings)[0]

        results = []
        for i, score in enumerate(scores):
            if score >= threshold:
                results.append((score.item(), self.sections[i]))

        results.sort(reverse=True, key=lambda x: x[0])
        
        for score, section in results:
            log_summary_query(query, section["title"], section["summary"], score, debug=True)

        if not results:
            # Fallback: find if query matches a known high-level section for overview
            matched_parent = next(
                (title for title in self.section_titles if query.lower() in title.lower() and " > " not in title),
                None
            )
            if matched_parent and overview_mode:
                children = [s for s in self.sections if s["title"].startswith(matched_parent + " > ")]
                if children:
                    summary = f"**{matched_parent}** contains the following topics:\n\n" + \
                            "\n".join(f"- {c['title'].split(' > ')[-1]}" for c in children)
                    return [{"title": matched_parent, "summary": summary, "related": [c["title"] for c in children]}]

            return [{"title": "Not found", "summary": "Sorry, I couldn't find a relevant section.", "related": []}]

        top_results = results[:top_k]
        return [
            {
                "title": section["title"],
                "summary": section["summary"],
                "content": section["content"],
                "related": self.get_related_topics(section["title"], count=5)
            }
            for _, section in top_results
        ]      
        
            
    
    
    def get_related_topics(self, title: str, count: int = 5):
        if title not in self.section_titles:
            return []

        idx = self.section_titles.index(title)
        target_embedding = self.section_embeddings[idx]

        similarities = util.pytorch_cos_sim(target_embedding, self.section_embeddings)[0]
        related = [
            (i, score.item()) for i, score in enumerate(similarities)
            if i != idx and score.item() > 0.4
        ]

        related.sort(key=lambda x: x[1], reverse=True)
        top_related = related[:count]
        return [self.sections[i]["title"] for i, _ in top_related]

