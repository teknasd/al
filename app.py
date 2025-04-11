import streamlit as st
import os
from utils import (
    init_bedrock_client,
    init_embeddings,
    init_llm,
    load_documents,
    init_vectorstore,
    create_rag_chain,
    get_response
)
from logger_config import logger

st.set_page_config(page_title="Customer Support RAG Chatbot", page_icon="🤖")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []

def add_log_message(message: str):
    """Add a log message to both logger and session state."""
    logger.info(message)
    st.session_state.log_messages.append(message)

def initialize_rag():
    """Initialize the RAG system."""
    try:
        # Create docs directory if it doesn't exist
        if not os.path.exists("docs"):
            add_log_message("docs directory not found, creating it")
            os.makedirs("docs")
            st.error("Please add your documentation files to the 'docs' directory and restart the application.")
            return False
        
        add_log_message("Initializing RAG components")
        
        # Initialize components
        bedrock_client = init_bedrock_client()
        embeddings = init_embeddings()
        llm = init_llm(bedrock_client)
        
        # Load and process documents
        add_log_message("Loading documents")
        documents = load_documents()
        if not documents:
            add_log_message("No documents found or loaded")
            st.error("No documents found in the 'docs' directory. Please add your documentation files.")
            return False
        
        # Initialize vector store and RAG chain
        add_log_message("Initializing vector store and RAG chain")
        vectorstore = init_vectorstore(documents, embeddings)
        st.session_state.rag_chain = create_rag_chain(vectorstore, llm)
        
        add_log_message("Successfully initialized RAG system")
        return True
    
    except Exception as e:
        error_msg = f"Error initializing RAG system: {str(e)}"
        add_log_message(error_msg)
        st.error(error_msg)
        return False

# Page title
st.title("Customer Support RAG Chatbot 🤖")

# Add a debug section in the sidebar
with st.sidebar:
    st.title("Debug Logs")
    if st.button("Clear Logs"):
        st.session_state.log_messages = []
    
    # Display logs in the sidebar
    for log in st.session_state.log_messages:
        st.text(log)

# Initialize RAG system if not already initialized
if st.session_state.rag_chain is None:
    add_log_message("Starting RAG system initialization")
    with st.spinner("Initializing the chatbot..."):
        if not initialize_rag():
            st.stop()

# Chat interface
st.write("Ask me anything about the documentation!")

# Display chat history
for i, (question, answer) in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)

# Chat input
if question := st.chat_input("Type your question here..."):
    add_log_message(f"Received question: {question}")
    
    # Add user message to chat
    with st.chat_message("user"):
        st.write(question)
    
    # Get bot response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                add_log_message("Generating response")
                response = get_response(
                    st.session_state.rag_chain,
                    question,
                    [(q, a) for q, a in st.session_state.chat_history]
                )
                
                # Display answer
                st.write(response["answer"])
                
                # Display sources in expander
                with st.expander("View Sources"):
                    for i, source in enumerate(response["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source)
                
                add_log_message("Successfully displayed response")
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            add_log_message(error_msg)
            st.error(error_msg)
            st.stop()
    
    # Update chat history
    st.session_state.chat_history.append((question, response["answer"]))
    add_log_message("Updated chat history") 