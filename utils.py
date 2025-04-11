import os
from typing import List, Dict
from dotenv import load_dotenv
import boto3
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_aws import ChatBedrock
import glob
from logger_config import logger

# Load environment variables
load_dotenv()

def init_bedrock_client():
    """Initialize AWS Bedrock client."""
    try:
        # First assume the IAM role
        sts_client = boto3.client('sts')
        logger.info("Attempting to assume IAM role")
        assumed_role_object = sts_client.assume_role(
            RoleArn=os.getenv('role_arn'),
            RoleSessionName="BedrokRAGSession"
        )
        
        # Get the temporary credentials
        credentials = assumed_role_object['Credentials']
        logger.info("Successfully assumed IAM role")
        
        # Create bedrock client with assumed role credentials
        client = boto3.client(
            service_name='bedrock-runtime',
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken'],
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )
        logger.info("Successfully initialized Bedrock client")
        return client
    except Exception as e:
        logger.error(f"Error initializing Bedrock client: {str(e)}")
        raise

def init_embeddings():
    """Initialize the embedding model."""
    try:
        logger.info("Initializing HuggingFace embeddings")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Successfully initialized embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}")
        raise

def init_llm(bedrock_client):
    """Initialize the LLM."""
    try:
        logger.info("Initializing Bedrock Chat model")
        llm = ChatBedrock(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            client=bedrock_client,
            model_kwargs={"temperature": 0}
        )
        logger.info("Successfully initialized LLM")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise

def load_documents(docs_dir: str = "docs"):
    """Load documents from the docs directory."""
    try:
        if not os.path.exists(docs_dir):
            logger.error(f"Directory {docs_dir} does not exist")
            return []
        
        # Get all PDF files in the directory
        pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {docs_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")
        
        # Load each PDF file individually
        documents = []
        for pdf_file in pdf_files:
            try:
                logger.info(f"Loading PDF file: {pdf_file}")
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Successfully loaded {pdf_file} with {len(docs)} pages")
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {str(e)}")
        
        if not documents:
            logger.warning("No documents were successfully loaded")
            return []
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(split_docs)} chunks")
        return split_docs
    except Exception as e:
        logger.error(f"Error in document loading process: {str(e)}")
        raise

def init_vectorstore(documents: List, embeddings):
    """Initialize and populate the vector store."""
    try:
        logger.info("Initializing vector store")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
        logger.info("Successfully initialized vector store")
        return vectorstore
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        raise

def create_rag_chain(vectorstore, llm):
    """Create the RAG chain."""
    try:
        logger.info("Creating RAG chain")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            verbose=True
        )
        logger.info("Successfully created RAG chain")
        return chain
    except Exception as e:
        logger.error(f"Error creating RAG chain: {str(e)}")
        raise

def get_response(chain, question: str, chat_history: List = None) -> Dict:
    """Get response from the RAG chain."""
    try:
        if chat_history is None:
            chat_history = []
        
        logger.info(f"Processing question: {question}")
        
        # Add system prompt to ensure responses are based only on retrieved context
        system_prompt = """
        You are a helpful and reliable customer support assistant for insurance company.
        Your are providing information about the insurance policies and claims.
        Respond only using the information available in the provided context. 
        If the answer is not explicitly found in the context provided, reply with: "I DONT KNOW" 
        Please do not guess, assume, or fabricate any information. 
        """
        
        response = chain(
            {"question": question, "chat_history": chat_history, "system_prompt": system_prompt}
        )
        
        logger.info("Successfully generated response")
        logger.debug(f"Response: {response['answer']}")
        logger.debug(f"Number of source documents: {len(response['source_documents'])}")
        
        return {
            "answer": response["answer"],
            "sources": [doc.page_content for doc in response["source_documents"]]
        }
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise 