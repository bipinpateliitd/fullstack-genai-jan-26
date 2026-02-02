# GenAI Agentic AI Syllabus Using LangChain

## Course Overview
This comprehensive course covers the fundamentals and advanced concepts of building Generative AI Agentic systems using LangChain and LangGraph. Students will learn to create intelligent, autonomous agents capable of reasoning, decision-making, and interacting with external tools and data sources.

**Duration**: 12 Weeks  
**Level**: Beginner to Advanced  
**Prerequisites**: Python programming, basic understanding of AI/ML concepts

---

## Module 1: Introduction to LangChain and Generative AI

### Learning Objectives
- Understand the fundamentals of Generative AI and Large Language Models (LLMs)
- Learn about the LangChain ecosystem and its components
- Set up development environment for LangChain projects
- Understand the difference between LangChain and LangGraph

### Topics Covered
- What is Generative AI and LLMs?
- Introduction to LangChain framework
- LangChain architecture and components
- Installing LangChain and dependencies
- Environment setup with `.env` files and API keys
- Overview of LangChain integrations (OpenAI, Anthropic, Google, Bedrock, etc.)
- LangChain vs LangGraph: When to use which?

### Hands-on Projects
- Setting up a LangChain development environment
- First LangChain application: Simple chat with an LLM
- Exploring different LLM providers

### Resources
- [LangChain Documentation](https://docs.langchain.com)
- [LangChain Installation Guide](https://docs.langchain.com/oss/python/langchain/install)

---

## Module 2: Chat Models and Prompt Engineering

### Learning Objectives
- Master different message types and formats
- Learn prompt engineering best practices
- Create reusable prompt templates
- Understand dynamic and context-aware prompting

### Topics Covered
- Chat models and message types (System, Human, AI messages)
- Message formats: Object-based vs Dictionary-based
- Prompt engineering fundamentals
- Creating and using prompt templates
- Variables and dynamic prompts
- Few-shot prompting techniques
- Chain-of-thought prompting
- Context engineering patterns
- Prompt versioning and testing with LangSmith

### Hands-on Projects
- Building a chatbot with custom prompts
- Creating reusable prompt templates
- Implementing few-shot learning examples
- Dynamic prompt generation based on user context

### Resources
- [Prompt Engineering Quickstart](https://docs.langchain.com/langsmith/prompt-engineering-quickstart)
- [Chat Models Documentation](https://docs.langchain.com/oss/python/langchain/models)

---

## Module 3: LangChain Expression Language (LCEL)

### Learning Objectives
- Understand LCEL syntax and composition
- Build complex chains using LCEL
- Learn about runnables and their properties
- Master chain composition patterns

### Topics Covered
- Introduction to LCEL
- Runnables and Runnable interfaces
- Chain composition with pipes (`|`)
- RunnablePassthrough and RunnableLambda
- Parallel execution with RunnableParallel
- Conditional logic in chains
- Error handling in chains
- Streaming with LCEL

### Hands-on Projects
- Building a simple LCEL chain
- Creating a multi-step processing pipeline
- Implementing parallel execution chains
- Error handling and retry mechanisms

### Resources
- [LCEL Documentation](https://docs.langchain.com/oss/python/langchain/component-architecture)

---

## Module 4: Memory and Conversation Management

### Learning Objectives
- Implement short-term and long-term memory
- Manage conversation history effectively
- Use different memory strategies
- Handle context window limitations

### Topics Covered
- Understanding memory in AI agents
- Short-term memory (conversation history)
- Long-term memory (cross-conversation context)
- Threads and sessions
- Message trimming and summarization
- Memory stores and persistence
- SQLite checkpointer for conversation history
- Managing multiple users and sessions
- Context window management strategies

### Hands-on Projects
- Building a chatbot with conversation memory
- Implementing SQLite-based memory persistence
- Multi-user, multi-session chat application
- Conversation summarization system

### Resources
- [Short-term Memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [Dynamic Cross-conversation Context](https://docs.langchain.com/oss/python/concepts/context)

---

## Module 5: Tools and Function Calling

### Learning Objectives
- Create custom tools for agents
- Understand tool calling mechanisms
- Integrate external APIs and services
- Handle tool execution and errors

### Topics Covered
- What are tools in LangChain?
- Tool calling fundamentals
- Creating custom tools with `@tool` decorator
- Tool schemas and input validation
- Built-in LangChain tools
- Integrating external APIs (Google Search, Wikipedia, etc.)
- Tool error handling and retries
- Dynamic tool selection
- Tool per agent pattern

### Hands-on Projects
- Creating custom calculator and search tools
- Building a weather information agent
- Integrating Google Maps API
- Database query tool implementation

### Resources
- [Tools Documentation](https://docs.langchain.com/oss/python/langchain/tools)
- [Tool Calling](https://docs.langchain.com/oss/python/langchain/models)

---

## Module 6: Retrieval Augmented Generation (RAG)

### Learning Objectives
- Understand RAG architecture and benefits
- Implement document loading and processing
- Create vector stores and embeddings
- Build end-to-end RAG applications

### Topics Covered
- Introduction to RAG
- Document loaders (PDF, CSV, Web, Google Drive, Notion, etc.)
- Text splitters and chunking strategies
  - RecursiveCharacterTextSplitter
  - CharacterTextSplitter
  - Semantic chunking
- Embedding models
  - OpenAI embeddings
  - HuggingFace embeddings
  - Bedrock embeddings
- Vector stores
  - FAISS
  - Chroma
  - Pinecone
  - Weaviate
- Retrieval strategies
- Semantic search and similarity
- Agentic RAG (agents that decide when to retrieve)

### Hands-on Projects
- Building a document Q&A system
- Creating a knowledge base chatbot
- Implementing semantic search
- Agentic RAG system with LangGraph

### Resources
- [RAG Documentation](https://docs.langchain.com/oss/python/langchain/component-architecture)
- [Agentic RAG Tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- [Text Splitters](https://docs.langchain.com/oss/python/integrations/splitters/index)

---

## Module 7: Introduction to LangGraph

### Learning Objectives
- Understand LangGraph architecture
- Learn state management concepts
- Build simple graphs with nodes and edges
- Implement conditional routing

### Topics Covered
- What is LangGraph?
- When to use LangGraph vs LangChain
- Graph primitives: State, Nodes, Edges
- StateGraph and state schemas
- Adding nodes and edges
- Conditional edges and routing
- START and END nodes
- Graph compilation and execution
- Visualizing graphs

### Hands-on Projects
- Building a simple state machine
- Creating a multi-step workflow
- Implementing conditional routing
- Graph visualization

### Resources
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Workflows and Agents](https://docs.langchain.com/oss/javascript/langgraph/workflows-agents)

---

## Module 8: Building Agents with LangChain and LangGraph

### Learning Objectives
- Create intelligent agents that can reason and act
- Implement ReAct (Reasoning + Acting) pattern
- Build custom agent architectures
- Handle agent loops and termination

### Topics Covered
- What are AI agents?
- Agent architectures and patterns
- ReAct (Reasoning and Acting) agents
- Creating agents with `create_react_agent`
- Agent state management
- Tool integration in agents
- Agent loops and max iterations
- Agent termination conditions
- Custom agent implementations with LangGraph
- Agent middleware and customization

### Hands-on Projects
- Building a research assistant agent
- Creating a customer support agent
- SQL query agent for database interaction
- Code generation and execution agent

### Resources
- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [Build a Custom RAG Agent](https://docs.langchain.com/oss/python/langgraph/agentic-rag)

---

## Module 9: Multi-Agent Systems

### Learning Objectives
- Design and implement multi-agent architectures
- Coordinate multiple specialized agents
- Implement agent handoffs and collaboration
- Build hierarchical agent systems

### Topics Covered
- Introduction to multi-agent systems
- Multi-agent patterns:
  - Subagents (agents as tools)
  - Handoffs (state-based routing)
  - Skills (on-demand context loading)
  - Router (classification-based routing)
  - Custom workflows
- Agent collaboration and communication
- State machine pattern for multi-step workflows
- Hierarchical agent architectures
- Agent orchestration strategies
- Deep Agents middleware

### Hands-on Projects
- Building a customer support system with specialized agents
- Creating a research team with multiple agents
- Implementing agent handoffs
- Hierarchical task delegation system

### Resources
- [Multi-agent Systems](https://docs.langchain.com/oss/python/langchain/multi-agent/index)
- [Subagents Pattern](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)

---

## Module 10: Persistence, Checkpointing, and Advanced Features

### Learning Objectives
- Implement persistence for long-running agents
- Use checkpointers for state management
- Enable human-in-the-loop workflows
- Implement time-travel debugging

### Topics Covered
- LangGraph persistence layer
- Checkpointers and checkpointing
  - MemorySaver
  - SqliteSaver
  - PostgresSaver
- Threads and thread management
- Human-in-the-loop patterns
- Interrupting and resuming execution
- Time-travel debugging
- State snapshots and replay
- Fault tolerance and error recovery
- Streaming outputs
  - Token streaming
  - Tool call streaming
  - Custom data streaming
- Async execution and callbacks

### Hands-on Projects
- Building a long-running agent with persistence
- Implementing human approval workflows
- Creating a debuggable agent with time-travel
- Streaming chatbot with real-time updates

### Resources
- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)

---

## Module 11: Testing, Evaluation, and Debugging

### Learning Objectives
- Test and evaluate LLM applications
- Use LangSmith for debugging and monitoring
- Create evaluation datasets
- Implement automated testing

### Topics Covered
- Introduction to LangSmith
- Tracing and debugging with LangSmith
- LangSmith Studio for local development
- Creating test datasets
- Evaluation approaches:
  - Offline evaluation (pre-deployment)
  - Online evaluation (production monitoring)
- LLM-as-judge evaluators
- RAG evaluation metrics:
  - Retrieval relevance
  - Answer correctness
  - Groundedness
- Custom evaluators
- A/B testing and experimentation
- Performance monitoring

### Hands-on Projects
- Setting up LangSmith tracing
- Creating evaluation datasets
- Implementing RAG evaluation
- Building custom evaluators
- A/B testing different prompts

### Resources
- [LangSmith Evaluation](https://docs.langchain.com/langsmith/evaluation)
- [Evaluate RAG Application](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)
- [LangSmith Studio](https://docs.langchain.com/oss/python/langchain/studio)

---

## Module 12: Production Deployment and Best Practices

### Learning Objectives
- Deploy agents to production
- Implement security best practices
- Optimize performance and costs
- Monitor production systems

### Topics Covered
- Deployment strategies
  - AWS Lambda deployment
  - Container-based deployment
  - Agent Server deployment
- LangSmith Deployment (formerly LangGraph Platform)
- Assistants, threads, and runs API
- Cron jobs and webhooks
- Security best practices:
  - API key management
  - Input validation and sanitization
  - Rate limiting
  - Access control
- Performance optimization:
  - Caching strategies
  - Batch processing
  - Parallel execution
- Cost optimization
- Production monitoring and alerting
- Error handling and logging
- Scaling considerations

### Hands-on Projects
- Deploying an agent to AWS Lambda
- Setting up production monitoring
- Implementing rate limiting
- Creating a production-ready chatbot

### Resources
- [Agent Server](https://docs.langchain.com/langsmith/deployment)
- [LangSmith Deployment](https://docs.langchain.com/langsmith/home)

---

## Capstone Project

### Project Options

1. **Intelligent Customer Support System**
   - Multi-agent system with specialized agents
   - RAG for knowledge base
   - Human-in-the-loop for escalations
   - Full production deployment

2. **Research Assistant Agent**
   - Web search and document retrieval
   - Multi-step reasoning
   - Report generation
   - Citation tracking

3. **Code Analysis and Generation System**
   - Code understanding and analysis
   - Automated code generation
   - Testing and validation
   - Documentation generation

4. **Personal AI Assistant**
   - Multi-modal capabilities
   - Memory and personalization
   - Task automation
   - Calendar and email integration

### Capstone Requirements
- Use LangChain and LangGraph
- Implement at least 3 modules from the course
- Include RAG or multi-agent architecture
- Implement persistence and memory
- Include evaluation and testing
- Deploy to production
- Create comprehensive documentation

---

## Assessment Methods

1. **Weekly Assignments** (40%)
   - Hands-on coding exercises
   - Mini-projects for each module

2. **Quizzes** (20%)
   - Conceptual understanding
   - Best practices
   - Architecture decisions

3. **Capstone Project** (40%)
   - Code quality and functionality
   - Architecture and design
   - Documentation
   - Presentation

---

## Recommended Tools and Technologies

### Required
- Python 3.9+
- LangChain
- LangGraph
- OpenAI API (or alternative LLM provider)
- Git and GitHub
- Jupyter Notebooks

### Optional but Recommended
- LangSmith account
- Docker
- AWS account (for deployment)
- Vector database (Pinecone, Weaviate, or Chroma)
- PostgreSQL (for production persistence)

---

## Additional Resources

### Official Documentation
- [LangChain Documentation](https://docs.langchain.com)
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangSmith Documentation](https://docs.langchain.com/langsmith/home)

### Community
- LangChain Discord
- LangChain GitHub Discussions
- Stack Overflow (langchain tag)

### Blogs and Tutorials
- LangChain Blog
- RAG From Scratch series
- LangChain YouTube channel

---

## Course Outcomes

By the end of this course, students will be able to:

1. ✅ Build production-ready AI agents using LangChain and LangGraph
2. ✅ Implement RAG systems for knowledge-intensive applications
3. ✅ Design and deploy multi-agent architectures
4. ✅ Create conversational AI with memory and context management
5. ✅ Integrate external tools and APIs into agents
6. ✅ Test, evaluate, and debug LLM applications
7. ✅ Deploy and monitor agents in production
8. ✅ Apply best practices for security, performance, and cost optimization
9. ✅ Build custom agent workflows with LangGraph
10. ✅ Understand when to use different patterns and architectures

---

## Prerequisites Checklist

Before starting this course, ensure you have:

- [ ] Strong Python programming skills
- [ ] Understanding of async/await in Python
- [ ] Basic knowledge of APIs and REST
- [ ] Familiarity with Git and version control
- [ ] Basic understanding of machine learning concepts
- [ ] Experience with Jupyter Notebooks
- [ ] API key for at least one LLM provider (OpenAI, Anthropic, etc.)

---

## Next Steps

1. Set up your development environment
2. Create accounts for LangSmith and your chosen LLM provider
3. Clone the course repository
4. Complete Module 1 setup exercises
5. Join the course community channels

**Happy Learning! 🚀**
