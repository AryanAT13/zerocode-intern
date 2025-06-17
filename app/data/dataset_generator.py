from datasets import Dataset
import pandas as pd
import random

# Seed finance topics
FINANCE_TOPICS = [
    "compound interest", "stock valuation", "retirement planning", 
    "tax optimization", "cryptocurrency risks", "mortgage types"
]

def generate_qa_pair(topic: str) -> dict:
    questions = [
        f"How does {topic} work?",
        f"What are the best strategies for {topic}?",
        f"Explain {topic} like I'm a beginner",
        f"What are common mistakes in {topic}?"
    ]
    
    answers = [
        f"{topic.capitalize()} involves [...] Key considerations include [...]",
        f"Top 3 strategies: 1) [...] 2) [...] 3) [...]",
        f"Imagine {topic} as [...] The core principle is [...]",
        f"Most people misunderstand [...] Avoid these 2 pitfalls: [...]"
    ]
    
    return {
        "question": random.choice(questions),
        "answer": random.choice(answers),
        "source": "synthetic",
        "topic": topic
    }

# Generate 550 samples (50 extra for safety)
dataset = [generate_qa_pair(random.choice(FINANCE_TOPICS)) for _ in range(550)]

# Convert to HuggingFace Dataset
hf_dataset = Dataset.from_pandas(pd.DataFrame(dataset))

# Save dataset
hf_dataset.save_to_disk("app/data/finance_qa")
print(f"Generated {len(dataset)} samples")