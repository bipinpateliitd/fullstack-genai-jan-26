"""
Tutorial 9: Agent with Custom Tools

This tutorial demonstrates:
- Creating custom tools for agents
- Agent initialization and execution
- Tool calling and execution
- Interactive agent responses
- Tool execution visualization

Run with: streamlit run 09_agent_with_tools.py
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import random
from datetime import datetime
import math

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agent with Tools - Tutorial 9",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Agent with Custom Tools")
st.caption("Tutorial 9: Building intelligent agents with tool-calling capabilities")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About Agents")
    st.markdown("""
    **Agents** can use tools to:
    - Perform calculations
    - Search information
    - Execute actions
    - Make decisions
    - Chain multiple tools
    
    **This tutorial demonstrates:**
    - Custom tool creation
    - Agent initialization
    - Tool execution
    - Multi-step reasoning
    - Interactive responses
    """)
    
    st.divider()
    
    st.header("⚙️ Agent Settings")
    model_name = st.selectbox(
        "Model",
        ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        index=1,
        help="GPT-4 models work better with tools"
    )
    
    temperature = st.slider(
        "Temperature",
        0.0, 1.0, 0.0, 0.1,
        help="Lower temperature for more reliable tool use"
    )
    
    max_iterations = st.slider(
        "Max Iterations",
        1, 10, 5,
        help="Maximum steps the agent can take"
    )
    
    st.divider()
    
    st.header("🛠️ Available Tools")
    st.markdown("""
    - **Calculator**: Math operations
    - **Random Number**: Generate random numbers
    - **Current Time**: Get current date/time
    - **String Reverser**: Reverse text
    - **Word Counter**: Count words in text
    """)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# Define custom tools
@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations. 
    Use this tool when you need to calculate numbers.
    
    Args:
        expression: A mathematical expression as a string (e.g., "2 + 2", "sqrt(16)", "10 * 5")
    
    Returns:
        The result of the calculation
    """
    try:
        # Safe evaluation of mathematical expressions
        # Add math functions to the namespace
        safe_dict = {
            "sqrt": math.sqrt,
            "pow": math.pow,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@tool
def random_number(min_val: int, max_val: int) -> str:
    """
    Generate a random number between min and max values.
    
    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
    
    Returns:
        A random number between min_val and max_val
    """
    number = random.randint(min_val, max_val)
    return f"Random number between {min_val} and {max_val}: {number}"

@tool
def current_time() -> str:
    """
    Get the current date and time.
    Use this when the user asks about the current time or date.
    
    Returns:
        Current date and time as a formatted string
    """
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def reverse_string(text: str) -> str:
    """
    Reverse a given text string.
    
    Args:
        text: The text to reverse
    
    Returns:
        The reversed text
    """
    return f"Reversed text: {text[::-1]}"

@tool
def word_counter(text: str) -> str:
    """
    Count the number of words in a text.
    
    Args:
        text: The text to analyze
    
    Returns:
        Number of words and characters in the text
    """
    words = len(text.split())
    chars = len(text)
    return f"Text analysis: {words} words, {chars} characters"

# Collect all tools
tools = [calculator, random_number, current_time, reverse_string, word_counter]

# Initialize LLM
@st.cache_resource
def get_llm(model: str, temp: float):
    return ChatOpenAI(model=model, temperature=temp)

llm = get_llm(model_name, temperature)

# Create agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to various tools.
    
When a user asks you to perform a task, think about which tool(s) would be most appropriate.
Use the tools when needed, and provide clear explanations of what you're doing.

Available tools:
- calculator: For mathematical calculations
- random_number: To generate random numbers
- current_time: To get the current date and time
- reverse_string: To reverse text
- word_counter: To count words and characters

Always explain your reasoning and the results clearly to the user."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=max_iterations,
    handle_parsing_errors=True
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat with Agent")
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                if "tools_used" in message and message["tools_used"]:
                    with st.expander("🛠️ Tools Used"):
                        for tool_info in message["tools_used"]:
                            st.markdown(f"**{tool_info}**")
    
    # Chat input
    if prompt_input := st.chat_input("Ask the agent to use tools..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt_input
        })
        
        with st.chat_message("user"):
            st.markdown(prompt_input)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Agent thinking..."):
                try:
                    # Prepare chat history
                    chat_history = []
                    for msg in st.session_state.messages[:-1]:  # Exclude current message
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))
                    
                    # Execute agent
                    response = agent_executor.invoke({
                        "input": prompt_input,
                        "chat_history": chat_history
                    })
                    
                    # Display response
                    st.markdown(response["output"])
                    
                    # Track tools used (simplified)
                    tools_used = []
                    if "intermediate_steps" in response:
                        for step in response["intermediate_steps"]:
                            if len(step) >= 2:
                                tool_name = step[0].tool
                                tools_used.append(tool_name)
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["output"],
                        "tools_used": tools_used
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with col2:
    st.header("💡 Example Prompts")
    
    example_prompts = [
        "What is the square root of 144?",
        "Generate a random number between 1 and 100",
        "What time is it now?",
        "Reverse the text: Hello World",
        "Count the words in: The quick brown fox jumps over the lazy dog",
        "Calculate 25 * 4 + 10",
        "What is pi times 2?",
        "Give me a random number between 50 and 100, then calculate its square root"
    ]
    
    for i, example in enumerate(example_prompts):
        if st.button(example, key=f"example_{i}"):
            st.session_state.temp_prompt = example
            st.rerun()
    
    # Handle example button clicks
    if "temp_prompt" in st.session_state:
        prompt_to_send = st.session_state.temp_prompt
        del st.session_state.temp_prompt
        
        # Add to messages and process
        st.session_state.messages.append({
            "role": "user",
            "content": prompt_to_send
        })
        st.rerun()

# Footer
st.divider()
st.caption("""
💡 **Tips:**
- Agents can chain multiple tools together
- Try complex requests that require multiple steps
- Lower temperature makes tool use more reliable
- GPT-4 models are better at using tools correctly
- Watch the "Tools Used" section to see agent reasoning
""")

# Info expander
with st.expander("🔍 Understanding Agents and Tools"):
    st.markdown("""
    ### What are Agents?
    
    **Agents** are LLMs that can:
    1. Reason about which tools to use
    2. Execute tools with appropriate inputs
    3. Interpret tool outputs
    4. Chain multiple tools together
    5. Provide final answers
    
    ### Agent Workflow:
    
    ```
    User Query
        ↓
    Agent Reasoning (which tool to use?)
        ↓
    Tool Selection
        ↓
    Tool Execution
        ↓
    Result Interpretation
        ↓
    Next Action Decision
        ↓
    Final Answer
    ```
    
    ### Creating Custom Tools:
    
    ```python
    from langchain_core.tools import tool
    
    @tool
    def my_custom_tool(input_param: str) -> str:
        '''
        Tool description that the agent reads.
        Be clear about what the tool does.
        
        Args:
            input_param: Description of the parameter
        
        Returns:
            Description of what is returned
        '''
        # Tool logic here
        result = process(input_param)
        return result
    ```
    
    ### Tool Best Practices:
    
    - ✅ **Clear descriptions**: Help the agent understand when to use the tool
    - ✅ **Type hints**: Specify parameter types
    - ✅ **Error handling**: Handle edge cases gracefully
    - ✅ **Focused purpose**: Each tool should do one thing well
    - ✅ **Descriptive names**: Use clear, action-oriented names
    
    ### Agent Types:
    
    **OpenAI Functions Agent**
    - Uses OpenAI's function calling
    - Most reliable for tool use
    - Works with GPT-3.5 and GPT-4
    
    **ReAct Agent**
    - Reasoning and Acting
    - More transparent reasoning
    - Works with any LLM
    
    **Structured Chat Agent**
    - For complex tool inputs
    - Better for multi-parameter tools
    
    ### Common Tool Categories:
    
    1. **Information Retrieval**
       - Web search
       - Database queries
       - API calls
    
    2. **Computation**
       - Math calculations
       - Data processing
       - Statistical analysis
    
    3. **Actions**
       - Send emails
       - Create files
       - Update databases
    
    4. **Analysis**
       - Text analysis
       - Image processing
       - Data validation
    
    ### Advanced Patterns:
    
    ```python
    # Tool with multiple parameters
    @tool
    def search_database(query: str, limit: int = 10) -> str:
        '''Search database with query and limit results'''
        results = db.search(query, limit=limit)
        return format_results(results)
    
    # Tool that returns structured data
    @tool
    def get_weather(city: str) -> dict:
        '''Get weather information for a city'''
        return {
            "temperature": 72,
            "condition": "sunny",
            "city": city
        }
    ```
    
    ### Debugging Agents:
    
    - Set `verbose=True` in AgentExecutor
    - Check intermediate_steps in response
    - Monitor tool calls and outputs
    - Test tools independently first
    - Use lower temperature for reliability
    """)
