from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict
load_dotenv(dotenv_path="/home/bipin/Documents/genai/g25-nov-hindi/fullstack-genai-jan-26/.env")
llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0.7,
        max_tokens=200,
    )


class Moviedetail(TypedDict):
    country:str
    year:int
    actor:list[str]
    rating:float
    buisness: int
    language: str
    
    
str_llm = llm.with_structured_output(Moviedetail)


response = str_llm.invoke("Pushpa")
