"""
Tutorial 5: Conversation Memory

This tutorial demonstrates:
- Managing conversation history with session state
- Context window handling
- Conversation summarization for long chats
- Memory management strategies

Run with: streamlit run 05_conversation_memory.py
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Conversation Memory - Tutorial 5",
    page_icon="🧠",
    layout="centered"
)

# Title and description
st.title("🧠 Conversation Memory with LangChain")
st.caption("Tutorial 5: Managing chat history and context")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tutorial demonstrates:
    - Session-based chat history
    - Message history management
    - Context window handling
    - Conversation summarization
    
    **Memory Strategies:**
    - Full history (simple)
    - Sliding window (last N messages)
    - Summarization (for long chats)
    """)
    
    st.divider()
    
    st.header("⚙️ Model Settings")
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
        step=0.1
    )
    
    st.divider()
    
    st.header("🧠 Memory Settings")
    
    memory_strategy = st.selectbox(
        "Memory Strategy",
        ["Full History", "Sliding Window", "Summarization"],
        help="How to manage conversation history"
    )
    
    if memory_strategy == "Sliding Window":
        window_size = st.slider(
            "Window Size (messages)",
            min_value=2,
            max_value=20,
            value=6,
            step=2,
            help="Number of recent messages to keep"
        )
    else:
        window_size = None
    
    if memory_strategy == "Summarization":
        summarize_threshold = st.slider(
            "Summarize after N messages",
            min_value=6,
            max_value=20,
            value=10,
            step=2,
            help="Summarize when history exceeds this length"
        )
    else:
        summarize_threshold = None
    
    st.divider()
    
    # Display memory stats
    if "messages" in st.session_state:
        st.header("📊 Memory Stats")
        st.metric("Total Messages", len(st.session_state.messages))
        
        if "summary" in st.session_state and st.session_state.summary:
            st.info("💡 Conversation has been summarized")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if "summary" in st.session_state:
            st.session_state.summary = None
        st.rerun()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# Initialize the ChatOpenAI model
@st.cache_resource
def get_llm(model: str, temp: float):
    """Initialize and cache the LLM"""
    return ChatOpenAI(model=model, temperature=temp)

llm = get_llm(model_name, temperature)

# Initialize chat history and summary
if "messages" not in st.session_state:
    st.session_state.messages = []

if "summary" not in st.session_state:
    st.session_state.summary = None

def get_conversation_context():
    """Get conversation context based on memory strategy"""
    messages = st.session_state.messages
    
    if memory_strategy == "Full History":
        # Return all messages
        return messages
    
    elif memory_strategy == "Sliding Window":
        # Return last N messages
        return messages[-window_size:] if len(messages) > window_size else messages
    
    elif memory_strategy == "Summarization":
        # If we have a summary, include it as context
        if st.session_state.summary:
            summary_msg = SystemMessage(content=f"Previous conversation summary: {st.session_state.summary}")
            # Return summary + recent messages
            recent_messages = messages[-6:]  # Keep last 6 messages
            return [summary_msg] + recent_messages
        else:
            return messages

async def summarize_conversation():
    """Summarize the conversation history"""
    if len(st.session_state.messages) < 4:
        return None
    
    # Create a summarization prompt
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following conversation concisely, capturing the key points and context:"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "Provide a brief summary of this conversation.")
    ])
    
    # Get summary
    messages_to_summarize = st.session_state.messages[:-2]  # Don't include last 2 messages
    formatted = summary_prompt.format_messages(history=messages_to_summarize)
    
    response = llm.invoke(formatted)
    return response.content

# Display current memory strategy
st.info(f"**Memory Strategy:** {memory_strategy}")

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
    
    # Check if we need to summarize (for summarization strategy)
    if memory_strategy == "Summarization" and summarize_threshold:
        if len(st.session_state.messages) >= summarize_threshold and not st.session_state.summary:
            with st.spinner("Summarizing conversation..."):
                import asyncio
                st.session_state.summary = asyncio.run(summarize_conversation())
                st.toast("✅ Conversation summarized!")
    
    # Get AI response
    with st.chat_message("assistant"):
        try:
            # Get conversation context based on strategy
            context = get_conversation_context()
            
            # Stream the response
            def stream_response():
                """Generator for streaming response"""
                for chunk in llm.stream(context):
                    if chunk.content:
                        yield chunk.content
            
            full_response = st.write_stream(stream_response())
            
            # Add AI response to chat history
            ai_message = AIMessage(content=full_response)
            st.session_state.messages.append(ai_message)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer with tips
st.divider()
st.caption("""
💡 **Tips:**
- **Full History**: Simple but can hit token limits with long conversations
- **Sliding Window**: Keeps recent context, good for most use cases
- **Summarization**: Best for very long conversations
- Watch the memory stats in the sidebar
- Next tutorial: Learn about RAG (Retrieval Augmented Generation)!
""")

# Additional info box
with st.expander("🔍 Understanding Conversation Memory"):
    st.markdown("""
    ### Why Memory Matters
    
    LLMs are stateless - they don't remember previous interactions. We need to:
    1. Store conversation history
    2. Include relevant history in each request
    3. Manage token limits
    
    ### Memory Strategies
    
    #### 1. Full History
    ```python
    # Simple: Include all messages
    messages = st.session_state.messages
    response = llm.invoke(messages)
    ```
    
    **Pros:** Complete context  
    **Cons:** Can exceed token limits
    
    #### 2. Sliding Window
    ```python
    # Keep last N messages
    recent_messages = messages[-window_size:]
    response = llm.invoke(recent_messages)
    ```
    
    **Pros:** Bounded memory, recent context  
    **Cons:** Loses older context
    
    #### 3. Summarization
    ```python
    # Summarize old messages, keep recent ones
    if len(messages) > threshold:
        summary = summarize(messages[:-6])
        context = [summary_message] + messages[-6:]
    else:
        context = messages
    ```
    
    **Pros:** Retains key information, bounded memory  
    **Cons:** More complex, extra LLM calls
    
    ### Token Management
    
    Most models have token limits:
    - GPT-3.5-turbo: 4,096 tokens
    - GPT-4o-mini: 128,000 tokens
    - GPT-4o: 128,000 tokens
    
    **Rule of thumb:** 1 token ≈ 4 characters
    
    ### Best Practices
    
    - ✅ Choose strategy based on use case
    - ✅ Monitor conversation length
    - ✅ Implement graceful degradation
    - ✅ Consider user experience
    - ✅ Test with long conversations
    
    ### Advanced Techniques
    
    - **Semantic Compression**: Keep semantically important messages
    - **Entity Extraction**: Track key entities across conversation
    - **Hybrid Approaches**: Combine multiple strategies
    - **Vector Store Memory**: Store embeddings of past conversations
    """)

# Show current context (for debugging)
with st.expander("👁️ View Current Context"):
    context = get_conversation_context()
    st.markdown(f"**Messages in context:** {len(context)}")
    
    if st.session_state.summary:
        st.markdown("**Summary:**")
        st.info(st.session_state.summary)
    
    st.markdown("**Context Messages:**")
    for i, msg in enumerate(context):
        role = "System" if isinstance(msg, SystemMessage) else ("User" if isinstance(msg, HumanMessage) else "Assistant")
        st.text(f"{i+1}. {role}: {msg.content[:100]}...")
