# Streamlit + LangChain Tutorial Series

A comprehensive tutorial series demonstrating Streamlit integration with LangChain, progressing from simple to advanced examples using the latest LangChain OpenAI patterns.

## 📚 Tutorial Overview

### Basic Level
1. **01_simple_chat.py** - Simple Streamlit Chat with ChatOpenAI
2. **02_streaming_chat.py** - Streaming Responses for Better UX
3. **03_prompt_templates.py** - Structured Prompts with Templates

### Intermediate Level
4. **04_lcel_chains.py** - LCEL Chains for Component Composition
5. **05_conversation_memory.py** - Conversation Memory Management
6. **06_basic_rag.py** - Basic RAG (Retrieval Augmented Generation)

### Advanced Level
7. **07_advanced_rag.py** - Advanced RAG with Source Attribution
8. **08_multi_model.py** - Multi-Model Comparison
9. **09_agent_with_tools.py** - Agent System with Custom Tools

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone or navigate to this directory**

```bash
cd streamlit-langchain-tutorials
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in this directory:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_actual_api_key_here
```

### Running the Tutorials

Each tutorial is a standalone Streamlit application. Run them using:

```bash
streamlit run 01_simple_chat.py
```

Replace `01_simple_chat.py` with any tutorial filename.

## 📖 Tutorial Details

### Tutorial 1: Simple Chat
**File**: `01_simple_chat.py`

Learn the basics:
- Setting up Streamlit chat interface
- Initializing ChatOpenAI
- Using `.invoke()` for responses
- Managing chat history with session state

**Key Concepts**: Basic chat, session state, invoke pattern

---

### Tutorial 2: Streaming Chat
**File**: `02_streaming_chat.py`

Improve user experience:
- Implementing streaming responses
- Using `.stream()` method
- Real-time token display with `st.write_stream()`
- Progressive output rendering

**Key Concepts**: Streaming, better UX, real-time responses

---

### Tutorial 3: Prompt Templates
**File**: `03_prompt_templates.py`

Structure your prompts:
- Using `ChatPromptTemplate`
- System and user message configuration
- Template variables and customization
- Dynamic prompt generation

**Key Concepts**: Prompt engineering, templates, structured prompts

---

### Tutorial 4: LCEL Chains
**File**: `04_lcel_chains.py`

Chain components together:
- LCEL pipe operator (`|`)
- Chaining prompts and models
- Output parsers
- Streaming chains

**Key Concepts**: LCEL, chains, composition

---

### Tutorial 5: Conversation Memory
**File**: `05_conversation_memory.py`

Maintain context:
- Session-based chat history
- Message history management
- Context window handling
- Conversation summarization

**Key Concepts**: Memory, context, chat history

---

### Tutorial 6: Basic RAG
**File**: `06_basic_rag.py`

Retrieval Augmented Generation:
- Document loading and processing
- Text splitting strategies
- Vector store creation (FAISS)
- Retrieval and generation
- File upload interface

**Key Concepts**: RAG, vector stores, document retrieval

---

### Tutorial 7: Advanced RAG
**File**: `07_advanced_rag.py`

Enhanced RAG capabilities:
- Multiple document types (PDF, TXT, CSV)
- Source tracking and citation
- Metadata-based retrieval
- Advanced retrieval strategies
- Source attribution in responses

**Key Concepts**: Advanced RAG, sources, metadata

---

### Tutorial 8: Multi-Model
**File**: `08_multi_model.py`

Compare different models:
- Model selection interface
- Multiple OpenAI models support
- Temperature and parameter controls
- Side-by-side comparison
- Performance metrics

**Key Concepts**: Model comparison, parameters, evaluation

---

### Tutorial 9: Agent with Tools
**File**: `09_agent_with_tools.py`

Build intelligent agents:
- Custom tool creation
- Agent initialization
- Tool calling and execution
- Interactive agent responses
- Tool execution visualization

**Key Concepts**: Agents, tools, autonomous behavior

## 🔧 Troubleshooting

### API Key Issues

If you see "OpenAI API key not found":
1. Ensure `.env` file exists in the tutorial directory
2. Verify the API key is correctly set in `.env`
3. Restart the Streamlit app

### Import Errors

If you encounter import errors:
```bash
pip install --upgrade -r requirements.txt
```

### Streamlit Issues

Clear Streamlit cache:
```bash
streamlit cache clear
```

## 📝 Learning Path

**Recommended Order**:
1. Start with Tutorial 1-3 to understand basics
2. Progress to Tutorial 4-6 for intermediate concepts
3. Complete Tutorial 7-9 for advanced techniques

Each tutorial builds on concepts from previous ones, so following the order is recommended for beginners.

## 🛠️ Technologies Used

- **Streamlit**: Web framework for data apps
- **LangChain**: Framework for LLM applications
- **LangChain-OpenAI**: OpenAI integration for LangChain
- **FAISS**: Vector store for similarity search
- **Python-dotenv**: Environment variable management

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

## 💡 Tips

- **API Costs**: Be mindful of API usage costs, especially with streaming and RAG
- **Model Selection**: Start with `gpt-3.5-turbo` for testing, upgrade to `gpt-4o` for better quality
- **Error Handling**: Each tutorial includes basic error handling for common issues
- **Customization**: Feel free to modify and extend these tutorials for your use cases

## 🤝 Contributing

These tutorials are designed for learning. Feel free to:
- Modify code for your needs
- Add new features
- Create additional tutorials
- Share improvements

## 📄 License

These tutorials are provided for educational purposes.

---

**Happy Learning! 🎉**

For questions or issues, please refer to the official documentation of the respective libraries.
