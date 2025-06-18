from sentence_transformers import SentenceTransformer

# Wrap the LLaMA3 embedder from HuggingFace
class LLaMA3Embedder:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)
