
import fitz  # PyMuPDF
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

#try different hugging face SentenceTransformer
#1. all-MiniLM-L12-v2
#Better accuracy than L6-v2, but slightly slower.
#Size: ~80MB
#Great for general-purpose semantic similarity.

#2. all-mpnet-base-v2
#One of the most accurate general-purpose English models in SentenceTransformers.
#Larger and slower, but significantly better embeddings.
#Good for semantic search and clustering.

class PDFKnowledgeBase:
    def __init__(self, pdf_path):
        
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
                
        
        self.sections = self.extract_sections(pdf_path)  # ⬅️ using structured sections
        # Semantic search embeddings
        self.section_titles = [s["title"] for s in self.sections]
        self.section_summaries = [s["summary"] for s in self.sections]
        #retrieve full original content, not just the summary.
        self.contents = [s["content"] for s in self.sections]
        
        # SentenceTransformer model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
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
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: b[1])  # top to bottom

            for block in blocks:
                text = block[4].strip()
                if len(text.split()) < 10 and text.istitle():  # naive title check
                    # Save previous section before starting new one
                    if title and content_buffer:
                        full_content = " ".join(content_buffer).strip()
                        #To avoid repeated summarization (which is slow), consider caching or summarizing only sections over a threshold length, e.g.: >20 or >50
                        if len(full_content.split()) > 50:  # filter out noise
                            section_data.append({
                            "title": title,
                            "content": full_content,
                            "summary": self.summarize(full_content)
                        })
                    title = text
                    content_buffer = []
                elif text and title:
                    content_buffer.append(text)

        
        # Add last section
        if title and content_buffer:
            full_content = " ".join(content_buffer).strip()
            if len(full_content.split()) > 50:
                section_data.append({
                    "title": title,
                    "content": full_content,
                    "summary": self.summarize(full_content)
                })

        return section_data   

    def summarize(self, text):
        if len(text) < 400:
            return text.strip()

        # Trust pipeline to truncate automatically
        try:
            summary = self.summarizer(text, max_len = min(200, int(len(text.split()) * 0.8)), min_length=60, do_sample=False)
            return summary[0]["summary_text"]
        except Exception as e:
            print("Summarization error:", e)
            return text[:300] + "..."

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
            return section["title"], section["summary"]
            
         # 🔁 Fuzzy fallback
        best_match, score = process.extractOne(query, self.section_titles)
        if score >= 70:  # Adjust fuzzy threshold
            index = self.section_titles.index(best_match)
            section = self.sections[index]
            return section["title"], section["summary"]            
        
            
    
    # #not use 
    # def search_by_title(self, query, top_k=1):
    #     # query_embedding = self.embedder.encode(query, convert_to_tensor=True)
    #     query_embedding = self.model.encode(query, convert_to_tensor=True)
    #     hits = util.semantic_search(query_embedding, self.section_embeddings, top_k=1)
    #     if not hits or not hits[0]:
    #         return "Not Found", "Sorry, I couldn’t find a matching section."
        
    #     best_hit = hits[0][0]
    #     section_index = best_hit["corpus_id"]
    #     match = self.sections[section_index]
    #     return match["title"], match["summary"]
    
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

