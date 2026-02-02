"""
Tutorial 4: LCEL Chains

This tutorial demonstrates:
- LCEL (LangChain Expression Language) for chaining components
- Using the pipe operator (|) to connect components
- Chaining prompts, models, and output parsers
- Streaming chains

Run with: streamlit run 04_lcel_chains.py
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LCEL Chains - Tutorial 4",
    page_icon="⛓️",
    layout="centered"
)

# Title and description
st.title("⛓️ LCEL Chains with LangChain")
st.caption("Tutorial 4: Component composition using LangChain Expression Language")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About LCEL")
    st.markdown("""
    **LCEL** (LangChain Expression Language) allows you to chain components together using the pipe operator `|`.
    
    **This tutorial demonstrates:**
    - Chaining prompts and models
    - Using `StrOutputParser`
    - Streaming chains
    - Multiple chain examples
    
    **Benefits:**
    - Clean, readable code
    - Easy composition
    - Automatic streaming support
    - Type safety
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
        step=0.1
    )
    
    st.divider()
    
    st.header("🔗 Chain Examples")
    chain_type = st.selectbox(
        "Select Chain Type",
        [
            "Simple Translation",
            "Joke Generator",
            "Code Explainer",
            "Summary Generator",
            "Custom Chain"
        ]
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
    return ChatOpenAI(model=model, temperature=temp)

llm = get_llm(model_name, temperature)

# Define different chain configurations
def get_chain(chain_type: str):
    """Create different LCEL chains based on selection"""
    
    if chain_type == "Simple Translation":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful translator. Translate the user's text to {target_language}."),
            ("user", "{text}")
        ])
        chain = prompt | llm | StrOutputParser()
        return chain, {"requires": ["target_language", "text"]}
    
    elif chain_type == "Joke Generator":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a funny comedian. Generate a hilarious joke about the given topic."),
            ("user", "Topic: {topic}")
        ])
        chain = prompt | llm | StrOutputParser()
        return chain, {"requires": ["topic"]}
    
    elif chain_type == "Code Explainer":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert programmer. Explain the following code in simple terms, including what it does and how it works."),
            ("user", "{code}")
        ])
        chain = prompt | llm | StrOutputParser()
        return chain, {"requires": ["code"]}
    
    elif chain_type == "Summary Generator":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a skilled summarizer. Create a concise summary of the following text in {num_sentences} sentences."),
            ("user", "{text}")
        ])
        chain = prompt | llm | StrOutputParser()
        return chain, {"requires": ["text", "num_sentences"]}
    
    else:  # Custom Chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_message}"),
            ("user", "{user_input}")
        ])
        chain = prompt | llm | StrOutputParser()
        return chain, {"requires": ["system_message", "user_input"]}

# Get the selected chain
chain, chain_config = get_chain(chain_type)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chain information
st.info(f"**Current Chain:** {chain_type}")

# Input section based on chain type
st.subheader("📝 Input")

inputs = {}

if chain_type == "Simple Translation":
    col1, col2 = st.columns([2, 1])
    with col1:
        inputs["text"] = st.text_area("Text to translate:", height=100)
    with col2:
        inputs["target_language"] = st.selectbox(
            "Target Language:",
            ["Spanish", "French", "German", "Italian", "Japanese", "Chinese", "Hindi"]
        )

elif chain_type == "Joke Generator":
    inputs["topic"] = st.text_input("Joke topic:")

elif chain_type == "Code Explainer":
    inputs["code"] = st.text_area(
        "Paste your code here:",
        height=150,
        placeholder="def hello():\n    print('Hello, World!')"
    )

elif chain_type == "Summary Generator":
    col1, col2 = st.columns([3, 1])
    with col1:
        inputs["text"] = st.text_area("Text to summarize:", height=150)
    with col2:
        inputs["num_sentences"] = st.number_input(
            "Number of sentences:",
            min_value=1,
            max_value=10,
            value=3
        )

else:  # Custom Chain
    inputs["system_message"] = st.text_area(
        "System Message:",
        value="You are a helpful AI assistant.",
        height=80
    )
    inputs["user_input"] = st.text_area("User Input:", height=100)

# Execute chain button
if st.button("🚀 Run Chain", type="primary"):
    # Validate inputs
    missing_inputs = [key for key in chain_config["requires"] if not inputs.get(key)]
    
    if missing_inputs:
        st.error(f"Please provide: {', '.join(missing_inputs)}")
    else:
        # Display input
        with st.chat_message("user"):
            if chain_type == "Simple Translation":
                st.markdown(f"Translate to {inputs['target_language']}: {inputs['text']}")
            elif chain_type == "Joke Generator":
                st.markdown(f"Generate a joke about: {inputs['topic']}")
            elif chain_type == "Code Explainer":
                st.markdown("Explain this code:")
                st.code(inputs['code'])
            elif chain_type == "Summary Generator":
                st.markdown(f"Summarize in {inputs['num_sentences']} sentences: {inputs['text']}")
            else:
                st.markdown(inputs['user_input'])
        
        # Execute chain with streaming
        with st.chat_message("assistant"):
            try:
                # Stream the chain output
                response = st.write_stream(chain.stream(inputs))
                
                # Add to history
                st.session_state.messages.append({
                    "input": inputs,
                    "output": response,
                    "chain_type": chain_type
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Display history
if st.session_state.messages:
    st.divider()
    st.subheader("📜 History")
    
    for idx, msg in enumerate(reversed(st.session_state.messages[-5:])):  # Show last 5
        with st.expander(f"Run {len(st.session_state.messages) - idx}: {msg['chain_type']}"):
            st.markdown("**Input:**")
            st.json(msg['input'])
            st.markdown("**Output:**")
            st.markdown(msg['output'])

# Footer with tips
st.divider()
st.caption("""
💡 **Tips:**
- LCEL chains use the pipe operator `|` to connect components
- Chains automatically support streaming with `.stream()`
- `StrOutputParser()` extracts string content from AI messages
- Try different chain types to see various use cases
- Next tutorial: Learn about conversation memory management!
""")

# Additional info box
with st.expander("🔍 Understanding LCEL Chains"):
    st.markdown("""
    ### What is LCEL?
    
    **LCEL** (LangChain Expression Language) is a declarative way to compose chains.
    
    ### Basic Chain Structure:
    
    ```python
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    
    # Define components
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}")
    ])
    model = ChatOpenAI()
    output_parser = StrOutputParser()
    
    # Chain them together with the pipe operator
    chain = prompt | model | output_parser
    
    # Invoke the chain
    result = chain.invoke({"input": "Hello!"})
    
    # Or stream the chain
    for chunk in chain.stream({"input": "Hello!"}):
        print(chunk, end="")
    ```
    
    ### How it Works:
    
    1. **Prompt Template** formats the input
    2. **Model** generates a response
    3. **Output Parser** extracts the string content
    
    ### Benefits of LCEL:
    
    - ✅ **Readable**: Clear, linear flow
    - ✅ **Composable**: Easy to add/remove components
    - ✅ **Streaming**: Automatic streaming support
    - ✅ **Type-safe**: Better IDE support
    - ✅ **Debuggable**: Easy to trace execution
    
    ### Common Patterns:
    
    ```python
    # Simple chain
    chain = prompt | model | parser
    
    # Multi-step chain
    chain = prompt1 | model | prompt2 | model | parser
    
    # With custom functions
    chain = prompt | model | custom_function | parser
    ```
    
    ### Output Parsers:
    
    - **StrOutputParser**: Extracts string content
    - **JsonOutputParser**: Parses JSON responses
    - **PydanticOutputParser**: Validates with Pydantic models
    - **ListOutputParser**: Parses lists
    """)
