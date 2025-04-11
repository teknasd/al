# RAG Customer Support Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using:
- ChromaDB for vector storage
- Amazon Bedrock (Claude 3 Sonnet) for LLM
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Streamlit for the user interface

## Setup

### Using Conda

1. Create and activate the Conda environment:
```bash
conda create -n rag-chatbot python=3.10 -y
conda activate rag-chatbot
pip install -r requirements.txt
```

### Configuration

1. Create a `.env` file with your AWS credentials:
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_region
```

2. Place your documentation files in the `docs` directory

### Running the Application

Start the chatbot with network access:
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then access the application in your browser:
1. If using SSH port forwarding:
   ```bash
   ssh -L 8501:localhost:8501 your-username@your-server-ip
   ```
   Then open: http://localhost:8501

2. Or access directly using server IP:
   ```
   http://your-server-ip:8501
   ```

Note: Make sure port 8501 is open in your server's firewall/security group. 