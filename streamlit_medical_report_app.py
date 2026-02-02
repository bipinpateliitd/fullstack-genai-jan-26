import streamlit as st
import base64
from typing import TypedDict, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv(dotenv_path="/home/bipin/Documents/genai/g25-nov-hindi/fullstack-genai-jan-26/.env")

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define data structures
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

# Encode image
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

# Extract data from image
def extract_medical_data(image_bytes):
    llm = get_llm()
    base64_image = encode_image(image_bytes)
    
    structured_model = llm.with_structured_output(KFTReport)
    
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
    
    result = structured_model.invoke([message])
    return result

# Main app
def main():
    st.set_page_config(
        page_title="Medical Report Analyzer",
        page_icon="🏥",
        layout="wide"
    )
    
    # Header
    st.title("🏥 Medical Report Analyzer")
    st.write("AI-Powered Lab Report Data Extraction using LangChain & OpenAI")
    
    st.divider()
    
    # File uploader
    st.subheader("📤 Upload Lab Report Image")
    uploaded_file = st.file_uploader(
        "Choose a medical report image (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        st.subheader("🖼️ Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        st.divider()
        
        # Analyze button
        if st.button("🔍 Analyze Report", type="primary"):
            with st.spinner("Extracting data from report..."):
                try:
                    # Convert image to bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format or 'PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Extract data
                    report_data = extract_medical_data(img_byte_arr)
                    
                    # Store in session state
                    st.session_state['report_data'] = report_data
                    st.success("✅ Data extracted successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Display results if available
    if 'report_data' in st.session_state:
        report_data = st.session_state['report_data']
        
        st.divider()
        
        # Patient Information Section
        st.subheader("👤 Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Patient Name:**", report_data.get('patient_name', 'N/A'))
            st.write("**Patient ID:**", report_data.get('patient_id', 'N/A'))
            st.write("**Laboratory:**", report_data.get('lab_name', 'N/A'))
        
        with col2:
            st.write("**Doctor:**", report_data.get('doctor', 'N/A'))
            st.write("**Report Date:**", report_data.get('report_date', 'N/A'))
        
        st.divider()
        
        # Test Results Section
        st.subheader("🔬 Test Results")
        
        if 'results' in report_data and report_data['results']:
            for test_key, test_data in report_data['results'].items():
                # Get values with proper null handling
                value = test_data.get('value')
                ref_low = test_data.get('ref_low')
                ref_high = test_data.get('ref_high')
                
                # Determine status only if all values are present
                if value is not None and ref_low is not None and ref_high is not None:
                    if value < ref_low:
                        status = "🔻 LOW"
                        status_color = "orange"
                    elif value > ref_high:
                        status = "🔺 HIGH"
                        status_color = "red"
                    else:
                        status = "✅ NORMAL"
                        status_color = "green"
                else:
                    status = "❓ UNKNOWN"
                    status_color = "gray"
                
                # Display test result
                with st.container():
                    st.write(f"### {test_data.get('test_name', test_key)}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        value_display = f"{value} {test_data.get('unit', '')}" if value is not None else "N/A"
                        st.metric(
                            label="Value",
                            value=value_display
                        )
                    
                    with col2:
                        if ref_low is not None and ref_high is not None:
                            range_display = f"{ref_low} - {ref_high} {test_data.get('unit', '')}"
                        else:
                            range_display = "N/A"
                        st.metric(
                            label="Reference Range",
                            value=range_display
                        )
                    
                    with col3:
                        st.metric(
                            label="Status",
                            value=status
                        )
                    
                    st.write("**Test Date:**", test_data.get('date', 'N/A'))
                    st.divider()
        else:
            st.warning("⚠️ No test results found in the report.")

if __name__ == "__main__":
    main()
