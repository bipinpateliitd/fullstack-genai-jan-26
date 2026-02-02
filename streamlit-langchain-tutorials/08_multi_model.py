"""
Tutorial 8: Multi-Model Comparison

This tutorial demonstrates:
- Model selection interface
- Support for multiple OpenAI models
- Temperature and parameter controls
- Side-by-side comparison mode
- Model performance metrics

Run with: streamlit run 08_multi_model.py
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Model Comparison - Tutorial 8",
    page_icon="🔄",
    layout="wide"
)

# Title and description
st.title("🔄 Multi-Model Comparison")
st.caption("Tutorial 8: Compare different LLM models side-by-side")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tutorial demonstrates:
    - Multiple model selection
    - Parameter customization
    - Side-by-side comparison
    - Performance metrics
    - Cost estimation
    
    **Compare:**
    - Different model capabilities
    - Response quality
    - Speed differences
    - Cost trade-offs
    """)
    
    st.divider()
    
    st.header("🎯 Comparison Mode")
    comparison_mode = st.radio(
        "Select Mode",
        ["Single Model", "Compare Two Models"],
        help="Choose single or comparison mode"
    )
    
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# Model configurations
MODEL_INFO = {
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "description": "Fast and cost-effective",
        "context": "4K tokens",
        "cost_per_1k": "$0.0015",
        "speed": "⚡⚡⚡"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "description": "Balanced performance",
        "context": "128K tokens",
        "cost_per_1k": "$0.00015",
        "speed": "⚡⚡"
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "Most capable model",
        "context": "128K tokens",
        "cost_per_1k": "$0.005",
        "speed": "⚡"
    }
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Model configuration UI
if comparison_mode == "Single Model":
    st.header("⚙️ Model Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_1 = st.selectbox(
            "Select Model",
            list(MODEL_INFO.keys()),
            format_func=lambda x: MODEL_INFO[x]["name"]
        )
        
        # Show model info
        info = MODEL_INFO[model_1]
        st.info(f"""
        **{info['name']}**  
        {info['description']}  
        Context: {info['context']} | Cost: {info['cost_per_1k']}/1K tokens | Speed: {info['speed']}
        """)
    
    with col2:
        temperature_1 = st.slider(
            "Temperature",
            0.0, 2.0, 0.7, 0.1,
            key="temp_1"
        )
        
        max_tokens_1 = st.number_input(
            "Max Tokens",
            100, 4000, 500,
            key="max_1"
        )
        
        system_msg = st.text_area(
            "System Message",
            "You are a helpful AI assistant.",
            height=80
        )
    
    # Create LLM
    llm_1 = ChatOpenAI(
        model=model_1,
        temperature=temperature_1,
        max_tokens=max_tokens_1
    )
    
    # Chat interface
    st.divider()
    st.header("💬 Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            start_time = time.time()
            
            messages = [SystemMessage(content=system_msg)] + st.session_state.messages
            response = st.write_stream(llm_1.stream(messages))
            
            elapsed_time = time.time() - start_time
            
            # Add response to history
            st.session_state.messages.append(AIMessage(content=response))
            
            # Show metrics
            st.caption(f"⏱️ Response time: {elapsed_time:.2f}s | Model: {MODEL_INFO[model_1]['name']}")

else:  # Compare Two Models
    st.header("⚙️ Model Comparison Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model A")
        model_1 = st.selectbox(
            "Select Model A",
            list(MODEL_INFO.keys()),
            format_func=lambda x: MODEL_INFO[x]["name"],
            key="model_a"
        )
        
        info_1 = MODEL_INFO[model_1]
        st.info(f"{info_1['description']} | {info_1['speed']}")
        
        temperature_1 = st.slider(
            "Temperature A",
            0.0, 2.0, 0.7, 0.1,
            key="temp_a"
        )
    
    with col2:
        st.subheader("🤖 Model B")
        model_2 = st.selectbox(
            "Select Model B",
            list(MODEL_INFO.keys()),
            format_func=lambda x: MODEL_INFO[x]["name"],
            key="model_b",
            index=2
        )
        
        info_2 = MODEL_INFO[model_2]
        st.info(f"{info_2['description']} | {info_2['speed']}")
        
        temperature_2 = st.slider(
            "Temperature B",
            0.0, 2.0, 0.7, 0.1,
            key="temp_b"
        )
    
    system_msg = st.text_area(
        "System Message (both models)",
        "You are a helpful AI assistant.",
        height=60
    )
    
    # Create LLMs
    llm_1 = ChatOpenAI(model=model_1, temperature=temperature_1)
    llm_2 = ChatOpenAI(model=model_2, temperature=temperature_2)
    
    # Comparison interface
    st.divider()
    st.header("💬 Side-by-Side Comparison")
    
    prompt = st.text_area(
        "Enter your prompt:",
        height=100,
        placeholder="Ask a question to compare model responses..."
    )
    
    if st.button("🚀 Compare Models", type="primary") and prompt:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🤖 {MODEL_INFO[model_1]['name']}")
            with st.spinner("Generating..."):
                start_time_1 = time.time()
                
                messages = [SystemMessage(content=system_msg), HumanMessage(content=prompt)]
                response_1 = st.write_stream(llm_1.stream(messages))
                
                elapsed_1 = time.time() - start_time_1
                
                st.caption(f"⏱️ {elapsed_1:.2f}s | 💰 {info_1['cost_per_1k']}/1K")
        
        with col2:
            st.subheader(f"🤖 {MODEL_INFO[model_2]['name']}")
            with st.spinner("Generating..."):
                start_time_2 = time.time()
                
                messages = [SystemMessage(content=system_msg), HumanMessage(content=prompt)]
                response_2 = st.write_stream(llm_2.stream(messages))
                
                elapsed_2 = time.time() - start_time_2
                
                st.caption(f"⏱️ {elapsed_2:.2f}s | 💰 {info_2['cost_per_1k']}/1K")
        
        # Comparison metrics
        st.divider()
        st.subheader("📊 Comparison Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Speed Winner",
                MODEL_INFO[model_1 if elapsed_1 < elapsed_2 else model_2]["name"],
                f"{abs(elapsed_1 - elapsed_2):.2f}s faster"
            )
        
        with col2:
            st.metric("Model A Length", f"{len(response_1)} chars")
            st.metric("Model B Length", f"{len(response_2)} chars")
        
        with col3:
            st.metric("Model A Time", f"{elapsed_1:.2f}s")
            st.metric("Model B Time", f"{elapsed_2:.2f}s")

# Footer
st.divider()
st.caption("""
💡 **Tips:**
- Compare models to find the best fit for your use case
- GPT-3.5-turbo: Fast and cheap, good for simple tasks
- GPT-4o-mini: Balanced performance and cost
- GPT-4o: Best quality, higher cost
- Temperature affects creativity vs consistency
- Next tutorial: Build agents with custom tools!
""")

# Info expander
with st.expander("🔍 Understanding Model Selection"):
    st.markdown("""
    ### Model Comparison Guide
    
    #### GPT-3.5-Turbo
    - ✅ **Best for**: Simple tasks, high volume, cost-sensitive applications
    - ✅ **Strengths**: Fast, cheap, good for straightforward queries
    - ❌ **Limitations**: Less capable with complex reasoning
    
    #### GPT-4o-mini
    - ✅ **Best for**: Balanced performance, most general use cases
    - ✅ **Strengths**: Good quality, reasonable cost, large context window
    - ❌ **Limitations**: Not as capable as GPT-4o
    
    #### GPT-4o
    - ✅ **Best for**: Complex tasks, high-quality output, critical applications
    - ✅ **Strengths**: Most capable, best reasoning, large context
    - ❌ **Limitations**: Slower, more expensive
    
    ### Temperature Parameter
    
    **Low (0.0 - 0.3)**
    - Deterministic, focused responses
    - Good for: Factual Q&A, code generation, data extraction
    
    **Medium (0.4 - 0.7)**
    - Balanced creativity and consistency
    - Good for: General chat, content writing, explanations
    
    **High (0.8 - 2.0)**
    - Creative, varied responses
    - Good for: Creative writing, brainstorming, diverse ideas
    
    ### Cost Optimization
    
    ```python
    # Use cheaper models for simple tasks
    if task_complexity == "simple":
        model = "gpt-3.5-turbo"
    elif task_complexity == "medium":
        model = "gpt-4o-mini"
    else:
        model = "gpt-4o"
    ```
    
    ### Performance Metrics
    
    - **Latency**: Time to first token
    - **Throughput**: Tokens per second
    - **Quality**: Response accuracy and relevance
    - **Cost**: Price per 1K tokens
    
    ### When to Use Each Model
    
    | Use Case | Recommended Model |
    |----------|------------------|
    | Customer support | GPT-3.5-turbo |
    | Content generation | GPT-4o-mini |
    | Code review | GPT-4o |
    | Simple Q&A | GPT-3.5-turbo |
    | Complex analysis | GPT-4o |
    | High volume | GPT-3.5-turbo |
    """)
