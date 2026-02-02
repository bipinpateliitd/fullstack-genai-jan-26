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

# Define structured output schema
class MovieDetail(TypedDict):
    country: str
    year: int
    actor: list[str]
    rating: float
    business: int
    language: str

# Create structured LLM
str_llm = llm.with_structured_output(MovieDetail)

# Streamlit App
st.title("🎬 Movie Information App")
st.write("Ask about any movie and get detailed information!")

# User input
movie_query = st.text_input("Enter a movie name or question about a movie:", placeholder="e.g., Pushpa, RRR, Inception")

# Submit button
if st.button("Get Movie Details"):
    if movie_query:
        # Check if API key is loaded
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ OPENAI_API_KEY not found in environment variables!")
            st.info("Please make sure your .env file contains OPENAI_API_KEY")
        else:
            with st.spinner("Fetching movie details..."):
                try:
                    # Get structured response from LLM
                    response = str_llm.invoke(movie_query)
                    
                    # Display success message
                    st.success("Movie details retrieved successfully!")
                    
                    # Display each field separately
                    st.subheader("Movie Details")
                    
                    # Country
                    st.markdown("**🌍 Country:**")
                    st.write(response.get("country", "N/A"))
                    
                    # Year
                    st.markdown("**📅 Year:**")
                    st.write(response.get("year", "N/A"))
                    
                    # Actors
                    st.markdown("**🎭 Actors:**")
                    actors = response.get("actor", [])
                    if actors:
                        for actor in actors:
                            st.write(f"• {actor}")
                    else:
                        st.write("N/A")
                    
                    # Rating
                    st.markdown("**⭐ Rating:**")
                    st.write(f"{response.get('rating', 'N/A')}/10")
                    
                    # Business
                    st.markdown("**💰 Business (in millions):**")
                    st.write(f"${response.get('business', 'N/A')}M")
                    
                    # Language
                    st.markdown("**🗣️ Language:**")
                    st.write(response.get("language", "N/A"))
                    
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
        st.warning("Please enter a movie name or question!")

# Footer
st.markdown("---")
st.markdown("*Powered by LangChain and OpenAI*")
