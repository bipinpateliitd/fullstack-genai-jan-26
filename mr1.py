import base64
from typing import TypedDict, Dict, List, Any
from typing_extensions import NotRequired  # Python <3.11
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


import os
from dotenv import load_dotenv
from typing import Optional, List, Literal
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path="/home/bipin/Documents/genai/g25-nov-hindi/fullstack-genai-jan-26/.env")
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
print("✅ Setup complete!")

class LabResult(TypedDict):
    test_name: str
    date: str
    value: float
    unit: str
    ref_low: float
    ref_high: float

class KFTReport(TypedDict):
    patient_name: str
    patient_id: str
    doctor: str
    report_date: str
    lab_name: str
    results: Dict[str, LabResult]

# Encode image (same as before)
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_path = "a1.png"
base64_image = encode_image(image_path)

structured_model = llm.with_structured_output(KFTReport)  # TypedDict works directly!

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": """Extract kidney function test data from this lab report image into exact TypedDict format.
            Parse ALL tests (Urea, Creatinine, eGFR, Calcium, etc.) with precise values, units, ref ranges.
            Dates in YYYY-MM-DD. Use exact table values."""
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        },
    ]
)

result: KFTReport = structured_model.invoke([message])
print(result)
# {'patient_name': 'Mohan Sharma', 'patient_id': 'MR No.: 230923003', 'doctor': 'Dr. S. K. Gupta', ...}