"""
Tutorial 3: Prompt Templates

This tutorial demonstrates:
- Using ChatPromptTemplate for structured prompts
- System and user message configuration
- Template variables and customization
- Dynamic prompt generation

Run with: streamlit run 03_prompt_templates.py
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Prompt Templates - Tutorial 3",
    page_icon="📝",
    layout="centered"
)

# Title and description
st.title("📝 Prompt Templates with LangChain")
st.caption("Tutorial 3: Structured prompts using ChatPromptTemplate")

# Sidebar with information and settings
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tutorial demonstrates:
    - `ChatPromptTemplate` usage
    - System message configuration
    - Template variables
    - Dynamic prompt generation
    
    **Benefits of Templates:**
    - Consistent prompt structure
    - Easy customization
    - Reusable prompt patterns
    - Better prompt engineering
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
    
    st.header("🎭 Persona Settings")
    
    # Predefined personas
    persona_options = {
        "Helpful Assistant": "You are a helpful AI assistant. Provide clear, accurate, and friendly responses.",
        "Expert Teacher": "You are an expert teacher. Explain concepts clearly with examples and encourage learning.",
        "Creative Writer": "You are a creative writer. Use vivid language, metaphors, and engaging storytelling.",
        "Technical Expert": "You are a technical expert. Provide detailed, precise, and technical explanations.",
        "Friendly Companion": "You are a friendly companion. Be warm, empathetic, and conversational.",
        "Custom": ""
    }
    
    selected_persona = st.selectbox(
        "Select Persona",
        list(persona_options.keys())
    )
    
    if selected_persona == "Custom":
        system_message = st.text_area(
            "Custom System Message",
            value="You are a helpful AI assistant.",
            height=100,
            help="Define the AI's personality and behavior"
        )
    else:
        system_message = persona_options[selected_persona]
        st.info(f"**System Message:**\n\n{system_message}")
    
    st.divider()
    
    st.header("🎨 Response Style")
    response_style = st.selectbox(
        "Response Format",
        ["Default", "Concise", "Detailed", "Step-by-step", "With Examples"]
    )
    
    style_instructions = {
        "Default": "",
        "Concise": "Keep your response brief and to the point.",
        "Detailed": "Provide a comprehensive and detailed response.",
        "Step-by-step": "Break down your response into clear, numbered steps.",
        "With Examples": "Include relevant examples to illustrate your points."
    }
    
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

# Create the prompt template
def create_prompt_template(system_msg: str, style: str):
    """Create a ChatPromptTemplate with system and user messages"""
    
    # Build the system message with style instructions
    full_system_message = system_msg
    if style_instructions[style]:
        full_system_message += f"\n\n{style_instructions[style]}"
    
    # Create the template
    template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(full_system_message),
        HumanMessagePromptTemplate.from_template("{user_input}")
    ])
    
    return template

# Get the current prompt template
prompt_template = create_prompt_template(system_message, response_style)

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
    
    # Get AI response using the prompt template
    with st.chat_message("assistant"):
        try:
            # Format the prompt with the user input
            formatted_messages = prompt_template.format_messages(user_input=prompt)
            
            # For conversation history, we need to include previous messages
            # Combine system message with conversation history
            all_messages = formatted_messages[:-1]  # System message
            all_messages.extend(st.session_state.messages)  # Previous conversation
            
            # Stream the response
            def stream_response():
                """Generator for streaming response"""
                for chunk in llm.stream(all_messages):
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
- Try different personas to see how the AI's behavior changes
- Experiment with response styles for different use cases
- System messages set the overall behavior and personality
- Templates make it easy to maintain consistent prompts
- Next tutorial: Learn about LCEL chains for component composition!
""")

# Additional info box
with st.expander("🔍 Understanding Prompt Templates"):
    st.markdown("""
    ### What are Prompt Templates?
    
    Prompt templates provide a structured way to create prompts with:
    - **System Messages**: Define AI behavior and personality
    - **User Messages**: The actual user input
    - **Variables**: Dynamic content insertion
    
    ### Basic Template Structure:
    
    ```python
    from langchain_core.prompts import ChatPromptTemplate
    
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{user_input}")
    ])
    
    # Format the template
    messages = template.format_messages(user_input="Hello!")
    ```
    
    ### Benefits:
    
    - ✅ **Consistency**: Same structure across all interactions
    - ✅ **Reusability**: Define once, use many times
    - ✅ **Maintainability**: Easy to update prompts
    - ✅ **Flexibility**: Support for variables and dynamic content
    - ✅ **Best Practices**: Encourages good prompt engineering
    
    ### Common Use Cases:
    
    1. **Role-based AI**: Define specific roles (teacher, expert, etc.)
    2. **Output Formatting**: Request specific response formats
    3. **Context Setting**: Provide background information
    4. **Constraint Setting**: Define what AI should/shouldn't do
    
    ### Template Variables:
    
    ```python
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a {role}. {instructions}"),
        ("user", "{user_input}")
    ])
    
    messages = template.format_messages(
        role="teacher",
        instructions="Explain concepts simply.",
        user_input="What is AI?"
    )
    ```
    """)

# Show current template
with st.expander("👁️ View Current Template"):
    st.markdown("### Current Prompt Template")
    st.code(f"""
System Message:
{system_message}

Response Style Instruction:
{style_instructions[response_style] if style_instructions[response_style] else "None"}

User Input:
{{user_input}}
    """, language="text")
