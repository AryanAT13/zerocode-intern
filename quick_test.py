# quick_test.py

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

def main():
    from app.data.dataset_generator import generate_qa_pair
    test_topics = [
        "basic investing",
        "compound interest",
        "stock valuation",
        "retirement planning",
        "tax optimization"
    ]
    for topic in test_topics:
        qa = generate_qa_pair(topic)
        print(f"Topic: {topic}")
        print("Q:", qa["question"])
        print("A:", qa["answer"])
        print("-" * 40)

if __name__ == "__main__":
    main()




