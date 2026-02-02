"""
Tutorial 1: Simple Streamlit Chat with ChatOpenAI

This tutorial demonstrates:
- Basic Streamlit chat interface setup
- ChatOpenAI initialization
- Using .invoke() for responses
- Managing chat history with session state

Run with: streamlit run 01_simple_chat.py
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Simple Chat - Tutorial 1",
    page_icon="💬",
    layout="centered"
)

# Title and description
st.title("💬 Simple Chat with LangChain")
st.caption("Tutorial 1: Basic chat interface using ChatOpenAI")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tutorial demonstrates:
    - Basic chat interface
    - ChatOpenAI initialization
    - `.invoke()` method
    - Session state management
    """)
    
    st.divider()
    
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Model",
        ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        index=0
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random"
    )
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# Initialize the ChatOpenAI model
@st.cache_resource
def get_llm(model: str, temp: float):
    """Initialize and cache the LLM"""
    return ChatOpenAI(
        model=model,
        temperature=temp
    )

llm = get_llm(model_name, temperature)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response using invoke()
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the model with the full conversation history
                response = llm.invoke(st.session_state.messages)
                
                # Display the response
                st.markdown(response.content)
                
                # Add AI response to chat history
                st.session_state.messages.append(response)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer with tips
st.divider()
st.caption("""
💡 **Tips:**
- The chat maintains conversation history using Streamlit's session state
- Try changing the model or temperature in the sidebar
- Use the `.invoke()` method for single, complete responses
- Next tutorial: Learn about streaming responses for better UX!
""")
