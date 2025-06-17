from app.models.llm_handler import FinanceRAG
import pandas as pd
from evaluate import load
import nltk

# Ensure required NLTK data is available
nltk.download("punkt", quiet=True)

# Initialize evaluation metrics
bleu = load("bleu")
rouge = load("rouge")

# Sample evaluation questions
TEST_QUESTIONS = [
    ("What's a Roth IRA?", "retirement"),
    ("How do dividends work?", "investing"),
    ("Best way to pay off credit card debt?", "personal_finance")
]


def evaluate():
    # Initialize the RAG handler
    model = FinanceRAG()
    results = []

    for question, category in TEST_QUESTIONS:
        # Base model response (without retrieval)
        inputs = model.tokenizer(question, return_tensors="pt")
        base_ids = model.llm.generate(**inputs, max_length=100)
        base_response = model.tokenizer.decode(base_ids[0], skip_special_tokens=True)

        # RAG model response (with retrieval)
        rag_response = model.generate(question)

        # Compute metrics
        bleu_score = bleu.compute(predictions=[rag_response], references=[[base_response]])['bleu']
        rouge_score = rouge.compute(predictions=[rag_response], references=[base_response])['rougeL']

        results.append({
            "question": question,
            "category": category,
            "base_response": base_response,
            "rag_response": rag_response,
            "bleu": bleu_score,
            "rouge": rouge_score
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation complete. See evaluation_results.csv")


if __name__ == "__main__":
    evaluate()
