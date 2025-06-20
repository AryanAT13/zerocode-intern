from app.models.llm_handler import FinanceRAG

if __name__ == "__main__":
    model = FinanceRAG()
    query = "How do I start investing with $1000?"
    answer = model.generate(query)
    print(answer)