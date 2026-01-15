# Fullstack GenAI - January 2026

A comprehensive LangChain learning repository featuring tutorials and examples for building AI-powered applications using OpenAI's language models.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Tutorials](#tutorials)
- [Contributing](#contributing)

## 🎯 Overview

This repository contains hands-on tutorials and examples for learning LangChain, a powerful framework for developing applications powered by language models. The project demonstrates:

- LangChain basics and model initialization
- Working with different message types (System, Human, AI)
- Building conversational AI systems
- Implementing role-based AI assistants
- Multi-language support (English and Hindi)

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12 or higher**
- **uv** - A fast Python package installer and resolver
  - Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Or via pip: `pip install uv`
- **OpenAI API Key** - Get one from [OpenAI Platform](https://platform.openai.com/)

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fullstack-genai-jan-26
```

### 2. Install Dependencies

This project uses `uv` for dependency management. Install all dependencies from `pyproject.toml`:

```bash
uv sync
```

This command will:
- Create a virtual environment (if not already present)
- Install all dependencies specified in `pyproject.toml`
- Lock the dependencies in `uv.lock`

### 3. Activate the Virtual Environment

```bash
# On Linux/Mac
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

## 🔐 Environment Setup

### Create a `.env` File

Create a `.env` file in the root directory of the project:

```bash
touch .env
```

### Add Your API Key

Open the `.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

**Important:** Never commit your `.env` file to version control. It's already included in `.gitignore`.

## 📁 Project Structure

```
fullstack-genai-jan-26/
├── README.md              # Project documentation
├── pyproject.toml         # Project dependencies and metadata
├── uv.lock               # Locked dependency versions
├── main.py               # Main Python script
├── step1.ipynb           # LangChain tutorial notebook
└── .env                  # Environment variables (create this)
```

## 🚀 Usage

### Running the Main Script

```bash
python main.py
```

### Running Jupyter Notebooks

Start Jupyter Lab or Jupyter Notebook:

```bash
# Using Jupyter Lab
jupyter lab

# Or using Jupyter Notebook
jupyter notebook
```

Then open `step1.ipynb` to explore the LangChain tutorials.

## 📚 Tutorials

### Step 1: LangChain Basics (`step1.ipynb`)

This notebook covers:

1. **Environment Setup**
   - Loading environment variables with `python-dotenv`
   - Initializing ChatOpenAI models

2. **Basic LLM Interaction**
   - Simple text generation
   - Accessing response metadata and token usage

3. **Message Types**
   - Using `SystemMessage`, `HumanMessage`, and `AIMessage`
   - Dictionary-based message format

4. **Building Role-Based AI**
   - Creating specialized AI assistants (e.g., health expert)
   - System prompts and constraints
   - Multi-language responses (Hindi support)

## 🛠️ Dependencies

The project uses the following main dependencies:

- **langchain** (>=1.2.3) - Core LangChain framework
- **langchain-openai** (>=1.1.7) - OpenAI integration for LangChain
- **python-dotenv** (>=1.2.1) - Environment variable management
- **ipykernel** (>=7.1.0) - Jupyter notebook support

All dependencies are managed through `pyproject.toml` and installed via `uv sync`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is for educational purposes.

## 🔗 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [UV Documentation](https://github.com/astral-sh/uv)

---

**Happy Learning! 🚀**