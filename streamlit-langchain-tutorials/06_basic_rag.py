"""
Tutorial 6: Basic RAG (Retrieval Augmented Generation)

This tutorial demonstrates:
- Document loading and processing
- Text splitting strategies
- Vector store creation with FAISS
- Retrieval and generation
- File upload interface

Run with: streamlit run 06_basic_rag.py
"""

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Basic RAG - Tutorial 6",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 Basic RAG with LangChain")
st.caption("Tutorial 6: Retrieval Augmented Generation")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About RAG")
    st.markdown("""
    **RAG** (Retrieval Augmented Generation) enhances LLM responses by:
    1. Retrieving relevant documents
    2. Providing them as context
    3. Generating informed responses
    
    **This tutorial demonstrates:**
    - Document loading
    - Text splitting
    - Vector store creation
    - Similarity search
    - Context-aware generation
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
        value=0.3,
        step=0.1,
        help="Lower temperature for factual responses"
    )
    
    st.divider()
    
    st.header("📄 Text Splitting")
    chunk_size = st.slider(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Size of text chunks"
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="Overlap between chunks"
    )
    
    st.divider()
    
    st.header("🔍 Retrieval")
    num_docs = st.slider(
        "Documents to Retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of relevant chunks to retrieve"
    )
    
    if st.button("Clear Vector Store"):
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        st.rerun()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# Initialize LLM and embeddings
@st.cache_resource
def get_llm(model: str, temp: float):
    return ChatOpenAI(model=model, temperature=temp)

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

llm = get_llm(model_name, temperature)
embeddings = get_embeddings()

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Documents")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a text file",
        type=["txt"],
        help="Upload a .txt file to create a knowledge base"
    )
    
    # Or paste text
    st.markdown("**Or paste text directly:**")
    pasted_text = st.text_area(
        "Paste your text here",
        height=200,
        placeholder="Enter or paste text content..."
    )
    
    if st.button("🔨 Create Vector Store", type="primary"):
        text_content = None
        
        # Get text from file or pasted content
        if uploaded_file:
            text_content = uploaded_file.read().decode("utf-8")
        elif pasted_text:
            text_content = pasted_text
        else:
            st.error("Please upload a file or paste text!")
            st.stop()
        
        with st.spinner("Processing documents..."):
            try:
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len
                )
                
                chunks = text_splitter.create_documents([text_content])
                
                st.info(f"Created {len(chunks)} chunks from the document")
                
                # Create vector store
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.documents_loaded = True
                
                st.success(f"✅ Vector store created with {len(chunks)} chunks!")
                
            except Exception as e:
                st.error(f"Error creating vector store: {str(e)}")

with col2:
    st.header("💬 Ask Questions")
    
    if not st.session_state.documents_loaded:
        st.info("👈 Upload documents first to enable Q&A")
    else:
        st.success("✅ Vector store ready!")
        
        # Question input
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What is this document about?"
        )
        
        if st.button("🔍 Get Answer") and question:
            with st.spinner("Searching and generating answer..."):
                try:
                    # Retrieve relevant documents
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": num_docs}
                    )
                    
                    retrieved_docs = retriever.get_relevant_documents(question)
                    
                    # Show retrieved documents
                    with st.expander("📄 Retrieved Documents"):
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.text(doc.page_content[:300] + "...")
                            st.divider()
                    
                    # Create RAG chain
                    template = """Answer the question based only on the following context:

Context:
{context}

Question: {question}

Answer: Provide a detailed answer based on the context above. If the answer cannot be found in the context, say "I cannot answer this based on the provided documents."
"""
                    
                    prompt = ChatPromptTemplate.from_template(template)
                    
                    # Format context
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    # Create RAG chain using LCEL
                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Get answer
                    st.markdown("### 🤖 Answer:")
                    
                    # Stream the answer
                    answer = st.write_stream(rag_chain.stream(question))
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer with tips
st.divider()
st.caption("""
💡 **Tips:**
- Upload documents or paste text to create a knowledge base
- Adjust chunk size and overlap for better retrieval
- Lower temperature gives more factual responses
- Increase retrieved documents for more context
- Next tutorial: Learn about advanced RAG with source attribution!
""")

# Additional info box
with st.expander("🔍 Understanding RAG"):
    st.markdown("""
    ### What is RAG?
    
    **RAG** (Retrieval Augmented Generation) combines:
    - **Retrieval**: Finding relevant information
    - **Augmentation**: Adding it to the prompt
    - **Generation**: Creating informed responses
    
    ### RAG Pipeline:
    
    ```
    1. Document Loading
       ↓
    2. Text Splitting (chunking)
       ↓
    3. Embedding Generation
       ↓
    4. Vector Store Creation
       ↓
    5. User Query
       ↓
    6. Similarity Search
       ↓
    7. Context Retrieval
       ↓
    8. Prompt Augmentation
       ↓
    9. LLM Generation
    ```
    
    ### Key Components:
    
    #### 1. Text Splitter
    ```python
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Size of each chunk
        chunk_overlap=50,    # Overlap between chunks
    )
    chunks = text_splitter.create_documents([text])
    ```
    
    #### 2. Embeddings
    ```python
    embeddings = OpenAIEmbeddings()
    # Converts text to vector representations
    ```
    
    #### 3. Vector Store
    ```python
    vectorstore = FAISS.from_documents(chunks, embeddings)
    # Stores vectors for similarity search
    ```
    
    #### 4. Retriever
    ```python
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Return top 3 matches
    )
    ```
    
    #### 5. RAG Chain
    ```python
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    ```
    
    ### Benefits of RAG:
    
    - ✅ **Up-to-date Information**: Use current documents
    - ✅ **Domain-specific**: Add specialized knowledge
    - ✅ **Factual**: Grounded in provided documents
    - ✅ **Transparent**: Can show sources
    - ✅ **Cost-effective**: No model fine-tuning needed
    
    ### Best Practices:
    
    - Choose appropriate chunk size (500-1000 chars)
    - Use overlap to maintain context (10-20%)
    - Retrieve enough documents (3-5 typically)
    - Lower temperature for factual responses
    - Always cite sources in production
    
    ### Common Use Cases:
    
    - 📚 Document Q&A systems
    - 🔍 Knowledge base search
    - 📊 Report analysis
    - 📖 Research assistance
    - 💼 Enterprise data querying
    """)
