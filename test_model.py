from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict
import os

load_dotenv(dotenv_path="/home/bipin/Documents/genai/g25-nov-hindi/langchain-learning/.env")

print(f"API Key loaded: {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"API Key (first 10 chars): {os.getenv('OPENAI_API_KEY')[:10]}...")

try:
    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0.7,
        max_tokens=200,
    )
    
    class MovieDetail(TypedDict):
        country: str
        year: int
        actor: list[str]
        rating: float
        business: int
        language: str
    
    str_llm = llm.with_structured_output(MovieDetail)
    
    print("\nTesting with 'Pushpa'...")
    response = str_llm.invoke("Pushpa")
    print("\nSuccess! Response:")
    print(response)
    
except Exception as e:
    print(f"\nError: {e}")
    print(f"Error type: {type(e).__name__}")
