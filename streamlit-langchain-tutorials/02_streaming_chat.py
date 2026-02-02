"""
Tutorial 2: Streaming Responses

This tutorial demonstrates:
- Implementing streaming responses with .stream()
- Real-time token display using st.write_stream()
- Better user experience with progressive output
- Handling streaming chunks

Run with: streamlit run 02_streaming_chat.py
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
    page_title="Streaming Chat - Tutorial 2",
    page_icon="⚡",
    layout="centered"
)

# Title and description
st.title("⚡ Streaming Chat with LangChain")
st.caption("Tutorial 2: Real-time streaming responses for better UX")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tutorial demonstrates:
    - Streaming responses with `.stream()`
    - Real-time token display
    - Progressive output rendering
    - Better user experience
    
    **Key Difference from Tutorial 1:**
    - Tutorial 1 uses `.invoke()` - waits for complete response
    - Tutorial 2 uses `.stream()` - displays tokens as they arrive
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
    
    streaming_enabled = st.checkbox(
        "Enable Streaming",
        value=True,
        help="Toggle to compare streaming vs non-streaming"
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
    
    # Get AI response
    with st.chat_message("assistant"):
        try:
            if streaming_enabled:
                # STREAMING MODE: Display tokens as they arrive
                # Create a generator function for streaming
                def stream_response():
                    """Generator that yields content from streaming chunks"""
                    for chunk in llm.stream(st.session_state.messages):
                        if chunk.content:
                            yield chunk.content
                
                # Use st.write_stream to display streaming content
                full_response = st.write_stream(stream_response())
                
            else:
                # NON-STREAMING MODE: Wait for complete response
                with st.spinner("Thinking..."):
                    response = llm.invoke(st.session_state.messages)
                    full_response = response.content
                    st.markdown(full_response)
            
            # Add AI response to chat history
            ai_message = AIMessage(content=full_response)
            st.session_state.messages.append(ai_message)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer with tips
st.divider()
st.caption("""
💡 **Tips:**
- **Streaming ON**: Tokens appear progressively as they're generated (better UX!)
- **Streaming OFF**: Complete response appears at once (like Tutorial 1)
- Toggle streaming in the sidebar to see the difference
- Streaming is especially beneficial for longer responses
- Next tutorial: Learn about prompt templates for structured prompts!
""")

# Additional info box
with st.expander("🔍 How Streaming Works"):
    st.markdown("""
    ### Streaming vs Non-Streaming
    
    **Non-Streaming (`.invoke()`):**
    ```python
    response = llm.invoke(messages)
    # Waits for complete response
    st.markdown(response.content)
    ```
    
    **Streaming (`.stream()`):**
    ```python
    def stream_response():
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content
    
    full_response = st.write_stream(stream_response())
    # Displays tokens as they arrive
    ```
    
    ### Benefits of Streaming:
    - ✅ Better perceived performance
    - ✅ Users see progress immediately
    - ✅ More engaging user experience
    - ✅ Especially useful for long responses
    
    ### When to Use Each:
    - **Streaming**: User-facing chat applications, long-form content
    - **Non-streaming**: Batch processing, when you need the full response at once
    """)
