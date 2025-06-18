
import json
import os
import fitz  # PyMuPDF
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
        self.section_titles = [s["title"] for s in self.sections]
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
    

    def extract_sections(self, path):
        doc = fitz.open(path)
        section_data = []
        title = None
        content_buffer = []

        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            blocks.sort(key=lambda b: b["bbox"][1])  # top to bottom

            for block in blocks:
                if block["type"] != 0:
                    continue  # skip images, tables, etc.
                text = block["lines"]
                block_text = " ".join([" ".join([span["text"] for span in line["spans"]]) for line in text]).strip()
                if not block_text:
                    continue

                #short title-like lines are treated as headers
                if len(block_text.split()) < 10 and block_text.istitle():
                    if title and content_buffer:
                        full_content = " ".join(content_buffer).strip()
                        if len(full_content.split()) > 50:
                            section_data.append({
                                "title": title,
                                "content": full_content,
                                "summary": self.summarize(full_content)
                            })
                            log_debug_all(title, full_content, self.summarize(full_content))
                            # if self.debug:
                            #     print(f"\n[EXTRACTED SECTION]")
                            #     print(f"Title : {title}")
                            #     print(f"Content:\n{full_content}")  #
                            #     print(f"summary:\n{self.summarize(full_content)}")  # 
                    title = block_text
                    content_buffer = []
                elif title:
                    # Filter out tabular content (e.g. lots of numbers or pipes or short lines)
                    if not self.looks_like_table(block_text):
                        content_buffer.append(block_text)

        
        if title and content_buffer:
            full_content = " ".join(content_buffer).strip()
            if len(full_content.split()) > 50:
                section_data.append({
                    "title": title,
                    "content": full_content,
                    "summary": self.summarize(full_content)
                })
                log_debug_all(title, full_content, self.summarize(full_content))
                # if self.debug:
                #     print(f"\n[EXTRACTED SECTION]")
                #     print(f"Title : {title}")
                #     print(f"Content:\n{full_content}")  #
                #     print(f"summary:\n{self.summarize(full_content)}")  # 
        # After parsing:
        if self.debug:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(section_data, f, ensure_ascii=False, indent=2)
        
        return section_data
    
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

    #Performs semantic search over section embedding , and Returns summarized best match
    #0.4 is good for BERT-style models.
    #fuzzywuzzy score: Scale is 0–100. Use 70+ for reasonable match.
    def search(self, query, top_k: int = 1, score_threshold: float = 0.4):        
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)[0]    
        # Check top score
        if hits and hits[0]["score"] >= score_threshold:
            top_hit = hits[0]
            section = self.sections[top_hit["corpus_id"]]
            return section["title"], section["content"]
            
         # 🔁 Fuzzy fallback
        best_match, score = process.extractOne(query, self.section_titles)
        if score >= 70:  # Adjust fuzzy threshold
            index = self.section_titles.index(best_match)
            section = self.sections[index]
            return section["title"], section["content"]            
        
            
    
    
    def get_related_topics(self, query, top_n=3):
        # query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.title_embeddings, top_k=top_n + 1)[0]
        
        # Remove exact match if it exists
        related_titles = []
        for hit in hits:
            idx = hit["corpus_id"]            
            title = self.section_titles[idx]
            if title.lower() != query.lower():
                related_titles.append(title)
            if len(related_titles) >= top_n:
                break

        return related_titles

