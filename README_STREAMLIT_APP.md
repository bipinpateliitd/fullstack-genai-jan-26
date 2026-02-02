# Medical Report Analyzer - Streamlit App

A simple Streamlit web application that extracts structured data from medical lab report images using LangChain and OpenAI.

## Features

- 📤 Upload lab report images (PNG, JPG, JPEG)
- 🔍 Automatic data extraction using AI
- 👤 Display patient information
- 🔬 Show test results with values and reference ranges
- ✅ Status indicators (Normal, High, Low)

## Installation

Install the required dependencies:

```bash
pip install streamlit langchain langchain-openai langchain-core python-dotenv Pillow openai
```

Or use the requirements file:

```bash
pip install -r requirements_streamlit.txt
```

## Setup

1. Make sure you have a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. The app references the `.env` file at:
   `/home/bipin/Documents/genai/g25-nov-hindi/fullstack-genai-jan-26/.env`

## Running the App

Run the Streamlit app with:

```bash
streamlit run streamlit_medical_report_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## How to Use

1. **Upload Image**: Click on the file uploader and select a medical lab report image
2. **Analyze**: Click the "🔍 Analyze Report" button
3. **View Results**: The extracted data will be displayed in separate sections:
   - Patient Information (name, ID, doctor, date, lab)
   - Test Results (values, reference ranges, status)

## Supported Report Types

Currently optimized for Kidney Function Test (KFT) reports, but can be adapted for other medical reports.

## Technology Stack

- **Streamlit**: Web framework
- **LangChain**: AI orchestration
- **OpenAI GPT-4o-mini**: Vision and structured output
- **Python-dotenv**: Environment variable management
- **Pillow**: Image processing
