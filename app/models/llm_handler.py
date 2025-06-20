import logging
import numpy as np
import faiss
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinanceRAG")

class FinanceRAG:
    def __init__(self, k: int = 3, temperature: float = 0.7, top_p: float = 0.9):
        
        self.k = k
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"Initializing RAG with k={k}, temp={temperature}, top_p={top_p}")

        
        self.llm = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-base",
            device_map=None  
        )
        self.llm.to("cpu")
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

        
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.dataset = load_from_disk("app/data/finance_qa")

        
        self._build_index()

    def _build_index(self):
        questions = self.dataset["question"]
        embeddings = self.embedder.encode(questions, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings, dtype=np.float32))
        logger.info(f"Built FAISS index with {self.index.ntotal} entries")

    def retrieve(self, query: str) -> list[dict]:
        q_emb = self.embedder.encode([query])
        _, idxs = self.index.search(np.array(q_emb), self.k)
        return [self.dataset[int(i)] for i in idxs[0]]

    def generate(self, query: str) -> str:
        contexts = [item["answer"] for item in self.retrieve(query)]
        prompt = (
            "Answer this finance question using only the context below:\n"
            f"Question: {query}\nContext:\n" +
            "\n".join(contexts) + "\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=150,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=1.2
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.debug(f"Generated: {text}")
        return text
