NOVA — AI Assistant
NOVA is a modern Streamlit AI assistant built with Gemini and Ollama, featuring conversational memory and Retrieval-Augmented Generation (RAG) for document-based question answering.
Live Demo
🔗 Open NOVA
Overview
NOVA combines conversational AI with a local document knowledge base.
Documents can be indexed into ChromaDB and later retrieved using semantic search, allowing NOVA to answer questions using relevant stored information.
The application supports both:
Gemini for cloud-based AI responses
Ollama for local AI responses
Features
🤖 Gemini and Ollama AI modes
🧠 Conversation memory
📚 RAG-based document question answering
🔎 Semantic document search
🗃️ Persistent ChromaDB vector storage
👤 Permanent admin knowledge base
📄 Temporary user document uploads
🧹 Separate document clearing
💬 Automatic follow-up questions
⚡ Streaming AI responses
📑 PDF, TXT, DOCX, CSV and XLSX support
🎨 Clean Apple-inspired interface
RAG
NOVA uses a local RAG pipeline powered by:
Qwen3 Embedding 0.6B — document and query embeddings
ChromaDB — vector storage and retrieval
Qwen 2.5 3B — local response generation
Document Flow
Document → Text Extraction → Chunking → Qwen3 Embedding 0.6B → ChromaDB → Semantic Retrieval → Qwen 2.5 3B → Answer
Knowledge Base
NOVA separates documents into two categories:
Admin Knowledge
Permanent documents added by the administrator. These remain available even after users clear their uploaded documents.
User Documents
Documents uploaded during a session for temporary use. These can be removed using Clear Documents.
Supported Documents
PDF
TXT
DOCX
CSV
XLSX
Tech Stack
Python
Streamlit
Ollama
Qwen 2.5 3B
Qwen3 Embedding 0.6B
ChromaDB
Google Gemini
PyPDF
python-docx
pandas
Installation
Clone the repository:
git clone https://github.com/kishorecodes404/ProCode-basic-chatbot.git
cd ProCode-basic-chatbot
Create a virtual environment:
py -m venv .venv
.\.venv\Scripts\Activate.ps1
Install dependencies:
pip install streamlit google-genai python-dotenv pypdf chromadb requests pandas python-docx
Ollama Setup
Install Ollama and pull the required models:
ollama pull qwen2.5:3b
ollama pull qwen3-embedding:0.6b
Make sure Ollama is running before using Ollama (Local) mode.
Gemini Setup
Create a .env file:
GEMINI_API_KEY=your_gemini_api_key_here
Never commit your .env file to GitHub.
Run
streamlit run app.py
NOVA will be available at:
http://localhost:8501
Version History
V1 — Basic Gemini chatbot
V2 — Conversational interface with memory
V3 — RAG document chatbot
V4 — Local Ollama RAG with Qwen embeddings and separate admin/user knowledge bases
Author
Kishore M S
