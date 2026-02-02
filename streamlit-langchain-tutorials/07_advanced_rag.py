"""
Tutorial 7: Advanced RAG with Source Attribution

This tutorial demonstrates:
- Multiple document type support (PDF, TXT, CSV)
- Source tracking and citation
- Metadata-based retrieval
- Advanced retrieval strategies
- Source display in UI

Run with: streamlit run 07_advanced_rag.py
"""

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Advanced RAG - Tutorial 7",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 Advanced RAG with Source Attribution")
st.caption("Tutorial 7: Multi-format documents with source tracking")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About Advanced RAG")
    st.markdown("""
    This tutorial adds:
    - **Multiple file formats** (TXT, PDF, CSV)
    - **Source attribution** in responses
    - **Metadata tracking**
    - **Advanced retrieval** strategies
    - **Source visualization**
    
    **Key Improvements:**
    - Track which document answers came from
    - Support various file types
    - Display source snippets
    - Better context understanding
    """)
    
    st.divider()
    
    st.header("⚙️ Model Settings")
    model_name = st.selectbox(
        "Model",
        ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        index=1
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )
    
    st.divider()
    
    st.header("📄 Processing Settings")
    chunk_size = st.slider("Chunk Size", 200, 1500, 800, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 300, 100, 50)
    num_docs = st.slider("Documents to Retrieve", 1, 8, 4)
    
    st.divider()
    
    if "vectorstore" in st.session_state and st.session_state.vectorstore:
        st.header("📊 Knowledge Base Stats")
        st.metric("Total Documents", len(st.session_state.get("uploaded_files", [])))
        st.metric("Total Chunks", st.session_state.vectorstore.index.ntotal)
    
    if st.button("🗑️ Clear All"):
        for key in ["vectorstore", "uploaded_files", "all_docs"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# Initialize components
@st.cache_resource
def get_llm(model: str, temp: float):
    return ChatOpenAI(model=model, temperature=temp)

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

llm = get_llm(model_name, temperature)
embeddings = get_embeddings()

# Initialize session state
for key in ["vectorstore", "uploaded_files", "all_docs"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "vectorstore" else []

# Main content
tab1, tab2 = st.tabs(["📤 Upload Documents", "💬 Ask Questions"])

with tab1:
    st.header("Upload Multiple Documents")
    
    uploaded_files = st.file_uploader(
        "Upload documents (TXT, PDF, CSV)",
        type=["txt", "pdf", "csv"],
        accept_multiple_files=True,
        help="Upload one or more documents to build your knowledge base"
    )
    
    if uploaded_files and st.button("🔨 Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            try:
                all_documents = []
                
                for uploaded_file in uploaded_files:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Load based on file type
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                    
                    if file_extension == ".txt":
                        loader = TextLoader(tmp_path)
                        docs = loader.load()
                    elif file_extension == ".pdf":
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                    elif file_extension == ".csv":
                        loader = CSVLoader(tmp_path)
                        docs = loader.load()
                    else:
                        st.warning(f"Unsupported file type: {file_extension}")
                        continue
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata["source_file"] = uploaded_file.name
                        doc.metadata["file_type"] = file_extension
                    
                    all_documents.extend(docs)
                    os.unlink(tmp_path)  # Clean up temp file
                
                st.info(f"Loaded {len(all_documents)} documents from {len(uploaded_files)} files")
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                splits = text_splitter.split_documents(all_documents)
                st.info(f"Created {len(splits)} chunks")
                
                # Create vector store
                vectorstore = FAISS.from_documents(splits, embeddings)
                
                # Save to session state
                st.session_state.vectorstore = vectorstore
                st.session_state.uploaded_files = [f.name for f in uploaded_files]
                st.session_state.all_docs = splits
                
                st.success(f"✅ Processed {len(uploaded_files)} files into {len(splits)} searchable chunks!")
                
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
    
    # Show uploaded files
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("📁 Loaded Documents")
        for i, filename in enumerate(st.session_state.uploaded_files, 1):
            st.text(f"{i}. {filename}")

with tab2:
    if not st.session_state.vectorstore:
        st.info("👈 Please upload and process documents first")
    else:
        st.header("Ask Questions with Source Attribution")
        
        question = st.text_input(
            "Your question:",
            placeholder="What information can you find in these documents?"
        )
        
        show_sources = st.checkbox("Show source documents", value=True)
        
        if st.button("🔍 Get Answer with Sources", type="primary") and question:
            with st.spinner("Searching documents and generating answer..."):
                try:
                    # Retrieve relevant documents
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": num_docs}
                    )
                    
                    retrieved_docs = retriever.get_relevant_documents(question)
                    
                    # Create RAG prompt with source attribution
                    template = """Answer the question based on the following context. 
After your answer, cite the sources you used by mentioning the source file names.

Context:
{context}

Question: {question}

Answer: Provide a detailed answer and then list the sources used.
Format your response as:

**Answer:**
[Your detailed answer here]

**Sources:**
- [Source file names]
"""
                    
                    prompt = ChatPromptTemplate.from_template(template)
                    
                    # Format context with source info
                    def format_docs_with_sources(docs):
                        formatted = []
                        for i, doc in enumerate(docs, 1):
                            source = doc.metadata.get("source_file", "Unknown")
                            content = doc.page_content
                            formatted.append(f"[Source: {source}]\n{content}")
                        return "\n\n---\n\n".join(formatted)
                    
                    # Create RAG chain
                    rag_chain = (
                        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Get answer
                    st.markdown("### 🤖 Answer with Sources:")
                    answer = st.write_stream(rag_chain.stream(question))
                    
                    # Show source documents
                    if show_sources:
                        st.divider()
                        st.markdown("### 📚 Retrieved Source Documents:")
                        
                        for i, doc in enumerate(retrieved_docs, 1):
                            source_file = doc.metadata.get("source_file", "Unknown")
                            file_type = doc.metadata.get("file_type", "unknown")
                            
                            with st.expander(f"📄 Source {i}: {source_file} ({file_type})"):
                                st.markdown("**Content:**")
                                st.text(doc.page_content)
                                
                                st.markdown("**Metadata:**")
                                st.json(doc.metadata)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.divider()
st.caption("""
💡 **Advanced RAG Features:**
- Upload multiple documents in different formats
- Automatic source tracking and attribution
- Metadata-based organization
- Source visualization for transparency
- Next tutorial: Multi-model comparison!
""")

# Info expander
with st.expander("🔍 Advanced RAG Concepts"):
    st.markdown("""
    ### Source Attribution
    
    Source attribution makes RAG systems:
    - **Transparent**: Users see where information comes from
    - **Trustworthy**: Can verify claims
    - **Debuggable**: Identify retrieval issues
    
    ### Implementation:
    
    ```python
    # Add metadata to documents
    for doc in docs:
        doc.metadata["source_file"] = filename
        doc.metadata["file_type"] = extension
    
    # Format context with sources
    def format_docs_with_sources(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata["source_file"]
            formatted.append(f"[Source: {source}]\\n{doc.page_content}")
        return "\\n\\n".join(formatted)
    ```
    
    ### Multi-Format Support:
    
    **Text Files (.txt)**
    - Simple and fast
    - Good for plain text documents
    
    **PDF Files (.pdf)**
    - Preserves formatting
    - Handles multi-page documents
    - Requires PyPDF loader
    
    **CSV Files (.csv)**
    - Structured data
    - Each row becomes a document
    - Good for tabular information
    
    ### Best Practices:
    
    - ✅ Always track source metadata
    - ✅ Include page numbers for PDFs
    - ✅ Show sources to users
    - ✅ Handle multiple file types
    - ✅ Validate retrieved sources
    
    ### Advanced Retrieval Strategies:
    
    1. **Hybrid Search**: Combine keyword + semantic search
    2. **Re-ranking**: Re-order results by relevance
    3. **Filtering**: Filter by metadata before retrieval
    4. **Multi-query**: Generate multiple queries for better coverage
    5. **Parent Document**: Retrieve larger context around matches
    """)
