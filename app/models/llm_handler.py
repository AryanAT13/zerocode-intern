from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
import faiss
import numpy as np
import torch

class FinanceRAG:
    def __init__(self):
        # Load model (run on CPU)
        self.llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load dataset
        self.dataset = load_from_disk("app/data/finance_qa")
        self._build_index()
    
    def _build_index(self):
        # Create embeddings index
        questions = [q for q in self.dataset['question']]
        embeddings = self.embedder.encode(questions, show_progress_bar=True)
        
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))
        self.questions = questions
    
    def retrieve(self, query: str, k=3) -> list:
        # Retrieve relevant context
        query_embed = self.embedder.encode([query])
        _, indices = self.index.search(query_embed, k)
        return [self.dataset[int(i)] for i in indices[0]]
    
    def generate(self, query: str) -> str:
        # RAG pipeline
        context = "\n".join([q['answer'] for q in self.retrieve(query)])
        prompt = f"""Answer this finance question using only the context below:
        Question: {query}
        Context: {context}
        Answer:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm.generate(**inputs, max_length=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)