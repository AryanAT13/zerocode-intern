import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nltk
nltk.download("punkt", quiet=True)


from app.models.llm_handler import FinanceRAG
import pandas as pd
from evaluate import load

def evaluate():
    
    bleu = load("bleu")
    rouge = load("rouge")
    bertscore = load("bertscore") 

  
    model = FinanceRAG()

    
    TEST_QUESTIONS = [
        ("What's a Roth IRA?", "retirement planning"),
        ("How do dividends work?", "stock valuation"),
        ("Best way to pay off credit card debt?", "personal finance")
    ]

    results = []
    for question, category in TEST_QUESTIONS:
       
        inputs = model.tokenizer(question, return_tensors="pt")
        base_ids = model.llm.generate(**inputs, max_length=100)
        base_response = model.tokenizer.decode(base_ids[0], skip_special_tokens=True)

       
        rag_response = model.generate(question)

       
        bleu_score = bleu.compute(predictions=[rag_response], references=[[base_response]])["bleu"]

      
        rouge_score = rouge.compute(predictions=[rag_response], references=[base_response])["rougeL"]

   
        bert = bertscore.compute(predictions=[rag_response], references=[base_response], lang="en")
        bert_f1 = bert["f1"][0]


        results.append({
            "question": question,
            "category": category,
            "base_response": base_response,
            "rag_response": rag_response,
            "bleu": round(bleu_score, 4),
            "rougeL": round(rouge_score, 4),
            "bertscore_f1": round(bert_f1, 4)  
        })


    df = pd.DataFrame(results)
    df.to_csv("evaluation_results_full.csv", index=False)

    print("✅ Evaluation complete. See evaluation_results_full.csv")

if __name__ == "__main__":
    evaluate()
