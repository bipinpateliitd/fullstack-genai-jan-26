import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict
import os

# Load environment variables - try multiple paths
env_paths = [
    "/home/bipin/Documents/genai/g25-nov-hindi/fullstack-genai-jan-26/.env",
    ".env"
]

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        break
else:
    load_dotenv()  # Try loading from default location

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.7,
    max_tokens=500,
)

# Define structured output schema for medical consultation
class MedicalAdvice(TypedDict):
    about_the_problem: str
    severity_level: str
    doctor_visit: str
    food_to_eat: list[str]
    food_to_avoid: list[str]
    medicine: list[str]

# Create structured LLM
str_llm = llm.with_structured_output(MedicalAdvice)

# Streamlit App
st.title("🏥 Medical Consultation App")
st.write("Ask about any health concern and get detailed medical advice!")

# Disclaimer
st.warning("⚠️ **Disclaimer**: This is an AI-powered tool for informational purposes only. Always consult a qualified healthcare professional for medical advice.")

# User input
medical_query = st.text_area(
    "Describe your health concern or symptoms:", 
    placeholder="e.g., I have a headache and fever, What should I do for cold and cough?",
    height=100
)

# Submit button
if st.button("Get Medical Advice"):
    if medical_query:
        # Check if API key is loaded
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ OPENAI_API_KEY not found in environment variables!")
            st.info("Please make sure your .env file contains OPENAI_API_KEY")
        else:
            with st.spinner("Analyzing your concern..."):
                try:
                    # Get structured response from LLM
                    response = str_llm.invoke(medical_query)
                    
                    # Display success message
                    st.success("Medical advice retrieved successfully!")
                    
                    # Display each field separately
                    st.subheader("Medical Consultation Results")
                    
                    # About the problem
                    st.markdown("### 📋 About the Problem")
                    st.write(response.get("about_the_problem", "N/A"))
                    
                    # Severity level
                    st.markdown("### ⚠️ Severity Level")
                    severity = response.get("severity_level", "N/A")
                    
                    # Color code severity
                    if "high" in severity.lower() or "severe" in severity.lower():
                        st.error(f"🔴 {severity}")
                    elif "moderate" in severity.lower() or "medium" in severity.lower():
                        st.warning(f"🟡 {severity}")
                    else:
                        st.info(f"🟢 {severity}")
                    
                    # Doctor visit
                    st.markdown("### 👨‍⚕️ Doctor Visit Required")
                    doctor_visit = response.get("doctor_visit", "N/A")
                    if "yes" in doctor_visit.lower():
                        st.error(f"✅ {doctor_visit}")
                        st.write("**Please consult a doctor as soon as possible.**")
                    else:
                        st.success(f"❌ {doctor_visit}")
                    
                    # Food to eat
                    st.markdown("### 🥗 Food to Eat")
                    foods_to_eat = response.get("food_to_eat", [])
                    if foods_to_eat:
                        for food in foods_to_eat:
                            st.write(f"✅ {food}")
                    else:
                        st.write("No specific recommendations")
                    
                    # Food to avoid
                    st.markdown("### 🚫 Food to Avoid")
                    foods_to_avoid = response.get("food_to_avoid", [])
                    if foods_to_avoid:
                        for food in foods_to_avoid:
                            st.write(f"❌ {food}")
                    else:
                        st.write("No specific restrictions")
                    
                    # Medicine
                    st.markdown("### 💊 Medicine Recommendations")
                    medicines = response.get("medicine", [])
                    if medicines:
                        st.info("**Note**: These are general recommendations. Please consult a doctor before taking any medication.")
                        for medicine in medicines:
                            st.write(f"💊 {medicine}")
                    else:
                        st.write("No specific medication recommended")
                    
                    # Show raw response in expander
                    with st.expander("View Raw Response"):
                        st.json(response)
                        
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ An error occurred: {error_msg}")
                    
                    # Provide specific guidance based on error type
                    if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                        st.warning("🔑 This looks like an API key issue. Please check:")
                        st.write("1. Your .env file contains OPENAI_API_KEY")
                        st.write("2. The API key is valid and active")
                        st.write("3. You have sufficient credits in your OpenAI account")
                    elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                        st.warning("🌐 This looks like a connection issue. Please check your internet connection.")
                    elif "model" in error_msg.lower():
                        st.warning("🤖 This looks like a model issue. The model 'gpt-4.1-nano' might not be available.")
                        st.info("Try changing the model to 'gpt-4o-mini' or 'gpt-3.5-turbo'")
                    
                    with st.expander("View Full Error Details"):
                        st.code(error_msg)
    else:
        st.warning("Please describe your health concern!")

# Footer
st.markdown("---")
st.markdown("*Powered by LangChain and OpenAI*")
st.markdown("*Always consult a qualified healthcare professional for medical advice*")
