Hey there! I’m Aryan, an aspiring AI/ML engineer, and this is my Finance Chatbot with Retrieval‑Augmented Generation (RAG). I built it from scratch using free tools to fine‑tune a small LLM on a custom dataset of finance Q&A, and exposed it via a slick API. Strap in—this README walks you through everything, from setup to demo, data schema to model card, evaluation to Dockerization.  

Ever wished you could ask a chatbot specific finance questions—like “How do I start investing with $1000?”—and get clear, context‑grounded advice? That’s exactly what this project does:

1. **Dataset Creation**  
   - 10 **manual** expert Q&A pairs on basic investing, retirement, tax strategies, crypto risk, and more  
   - 540 **Gemini‑generated** Q&A pairs covering 6 finance topics  
   - Total: **550** high‑quality QA pairs  

2. **RAG Pipeline**  
   - **Embed** all questions with **all‑MiniLM‑L6‑v2**  
   - **Index** with **FAISS** (CPU)  
   - **Retrieve** top‑3 relevant answers for any query  
   - **Generate** a final response via **google/flan‑t5‑base**  

3. **API**  
   - FastAPI POST `/chat` endpoint  
   - Stateless: returns a new or existing `conversation_id`  

4. **Evaluation**  
   - Automated: BLEU, ROUGE‑L, BERTScore on 3 benchmark prompts  
   - Manual rubric: Accuracy / Completeness / Clarity / Relevance  

5. **Containerization**  
   - CPU‑only Docker image  
   - One‑command build & run  

---

##  Setup

### Prerequisites

- **Python 3.11+**  
- **Docker 20+** (optional but recommended)  
- Git (for cloning)  

1. Clone & venv

```bash
git clone https://github.com/<your-handle>/zerocode-ml-assignment.git
cd zerocode-ml-assignment
python3 -m venv .venv
source .venv/bin/activate

2. Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt


3. Environment Variables
Create a .env in the project root:

GEMINI_API_KEY=your-google-gemini-key



Data Schema (app/data/finance_qa/)

Each QA pair is a JSON record with:

Field	    Type	      Description
question	str	     The user’s finance question
answer	    str      The expert/Gemini‑generated answer
source	    str	     "manual" or "gemini"
topic	    str	     Finance topic (e.g. "compound interest")






On disk, we use Hugging Face’s datasets format. You can load it via:

from datasets import load_from_disk
ds = load_from_disk("app/data/finance_qa")
print(ds[0])



Model Card
1. Embedding Model
Name: sentence-transformers/all-MiniLM-L6-v2

Size: ~80 MB

Use: Generate 384‑dim embeddings for question retrieval via FAISS

2. Vector Store
Library: FAISS (CPU)

Index type: IndexFlatL2 (brute‑force L2 search)

Entries: 550 questions × 384 dims

3. Language Model (RAG)
Backbone: google/flan-t5-base (~1 GB)

Decoding:

max_new_tokens=150

temperature=0.7, top_p=0.9, repetition_penalty=1.2


Sample API & cURL Requests
Start your server:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload



New Conversationcurl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"How does compound interest work?"}'



Response:
{
  "response": "Compound interest means your interest earns interest... (etc.)",
  "conversation_id": "b1a7b6f2-..."
}


Continue Conversation
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message":"What about monthly compounding?",
    "conversation_id":"b1a7b6f2-..."
  }'




Evaluation
Run automated metrics:


python tests/run_evaluation.py
Inside Docker:

docker build -t finance-bot .
docker run --rm -v $(pwd):/app --env-file .env finance-bot python tests/run_evaluation.py

This produces evaluation_results_full.csv



Dockerization

docker build -t finance-bot .

Run (quick test)
docker run --rm -v $(pwd):/app --env-file .env finance-bot

Run API in Docker
docker run --rm -p 8000:8000 --env-file .env finance-bot uvicorn app.main:app --host 0.0.0.0
