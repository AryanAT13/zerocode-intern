import os, time, random, pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from tqdm import tqdm
from google.api_core.exceptions import GoogleAPICallError
from google.generativeai.types import GenerationConfig

import google.generativeai as genai



load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")
genai.configure(api_key=GEMINI_API_KEY)



SYNTHETIC_TOPICS = [
    "compound interest",
    "stock valuation",
    "retirement planning",
    "tax optimization",
    "cryptocurrency risks",
    "mortgage types"
]
MANUAL_QA_PAIRS = [
    {
        "question": "How do I start investing with $1000?",
        "answer": "Open a zero-fee brokerage account like Zerodha or Groww, split your $1000 between a broad-market ETF and a bond fund, and automate monthly SIPs to grow your portfolio steadily while reinvesting dividends and rebalancing annually.",
        "source": "manual",
        "topic": "basic investing"
    },
    {
        "question": "Best way to start investing with $1000 safely?",
        "answer": "Begin by building a small emergency fund, then allocate 70 of your $1000 to index funds and 30% to bonds, automate your investments monthly, and rebalance each year to minimize risk and build wealth safely.",
        "source": "manual",
        "topic": "retirement planning"
    },
    {
        "question": "How can a beginner invest $1000 wisely?",
        "answer": "Use a robo-advisor for fractional investing, diversify across U.S. and global ETFs, stick to low-cost platforms with minimal fees, and reinvest dividends to harness long-term compounding.",
        "source": "manual",
        "topic": "stock valuation"
    },
    {
        "question": "What are smart ways to deploy a $1000 investment?",
        "answer": "Avoid high-fee mutual funds, choose low-cost index ETFs like Nifty 50 or S&P 500, invest with a long-term mindset of at least 5 years, and set up automated SIPs to stay consistent.",
        "source": "manual",
        "topic": "tax optimization"
    },
    {
        "question": "What should I do with $1000 to grow wealth slowly?",
        "answer": "Keep a small emergency reserve in a high-yield savings account and invest the rest in a diversified ETF with monthly SIPs, rebalancing once a year to grow wealth gradually and steadily.",
        "source": "manual",
        "topic": "compound interest"
    },
    {
        "question": "How do I invest my first $1000 for retirement?",
        "answer": "Research retirement schemes like the NPS, open a retirement account, set up monthly SIPs, track your performance, and use available tax deductions to enhance long-term retirement savings.",
        "source": "manual",
        "topic": "retirement planning"
    },
    {
        "question": "Where to put $1000 for long-term growth?",
        "answer": "Use a zero-commission brokerage app to invest in a total market ETF, automate monthly investments, and keep a small cash buffer to stay invested and disciplined for long-term returns.",
        "source": "manual",
        "topic": "mortgage types"
    },
    {
        "question": "Can I invest $1000 in crypto and stay safe?",
        "answer": "Limit crypto exposure to 5/10 of your $1000 only if you understand the risks, keep the rest in diversified ETFs, use dollar-cost averaging, and steer clear of high-volatility meme coins.",
        "source": "manual",
        "topic": "cryptocurrency risks"
    },
    {
        "question": "How can I use $1000 to invest tax efficiently?",
        "answer": "Invest through tax-advantaged vehicles like a Roth IRA or ELSS, track all transactions, use tax-loss harvesting to reduce gains, and reinvest dividends in a tax-smart way.",
        "source": "manual",
        "topic": "tax optimization"
    },
    {
        "question": "How do I start a compounding portfolio with $1000?",
        "answer": "Choose low-cost index funds or ETFs, reinvest all returns, automate monthly contributions, and leave your earnings untouched to let compounding grow your wealth over time.",
        "source": "manual",
        "topic": "compound interest"
    }
]



PROMPT_TEMPLATES = [
    "List 3 bullet points for: {question}",
    "Provide 4 bullet points for: {question}",
    "As a finance expert, give 5 bullet points on: {question}"
]


def generate_with_gemini(prompt: str) -> str:
    backoff = 1
    model = genai.GenerativeModel("gemini-2.0-flash")
    generation_config = GenerationConfig(  
        temperature=0.7,
        max_output_tokens=150
    )
    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,  # Pass prompt directly
                generation_config=generation_config 
            )
            return response.text.strip()
        except GoogleAPICallError:
            if attempt < 2:
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                raise



def generate_qa_pair(topic: str) -> dict:
    if topic == "basic investing":
        return random.choice(MANUAL_QA_PAIRS)
    question = random.choice([
        f"How does {topic} work?",
        f"What are the best strategies for {topic}?",
        f"Explain {topic} like I'm a beginner",
        f"What are common mistakes in {topic}?"
    ])
    prompt = random.choice(PROMPT_TEMPLATES).format(question=question)
    answer = generate_with_gemini(prompt + "\n\n• ")
    return {
        "question": question,
        "answer": answer,
        "source": "gemini",
        "topic": topic
    }


if __name__ == "__main__":
    print("🔄 Building dataset… 10 manual + 540 Gemini‐generated = 550 total QA pairs")
    dataset = MANUAL_QA_PAIRS.copy()
    
    for _ in tqdm(range(5), desc="Generating with Gemini"):
        topic = random.choice(SYNTHETIC_TOPICS)
        dataset.append(generate_qa_pair(topic))

    df = pd.DataFrame(dataset)
    ds = Dataset.from_pandas(df)
    ds.save_to_disk("app/data/finance_qa")
    print(f"\n✅ Saved {len(dataset)} QA pairs to app/data/finance_qa")
