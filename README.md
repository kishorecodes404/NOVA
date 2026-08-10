# NOVA — V3: RAG Document Chatbot

NOVA is a Streamlit AI chatbot powered by Google Gemini. Version 3 adds Retrieval-Augmented Generation (RAG), allowing users to upload PDF documents and ask questions based on their contents.

## Features

- Gemini-powered conversational chatbot
- Conversation memory using Streamlit session state
- PDF document upload
- PDF text extraction with PyPDF
- Text chunking for efficient retrieval
- Gemini embeddings for semantic search
- ChromaDB vector database
- RAG-based answers grounded in uploaded documents
- PDF page source references
- Clear Chat control
- Clear Documents control
- Clean Apple-inspired Streamlit interface

## Tech Stack

- Python
- Streamlit
- Google GenAI SDK
- Gemini 3.6 Flash
- Gemini Embedding 2
- ChromaDB
- PyPDF
- python-dotenv

## Project Structure

```text
NOVA/
│
├── app.py
├── rag.py
├── .env
├── .gitignore
├── README.md
│
├── .streamlit/
│   └── config.toml
│
├── chroma_db/          # Created automatically; ignored by Git
└── __pycache__/        # Generated automatically; ignored by Git
```

## Installation

Clone the repository:

```bash
git clone https://github.com/kishorecodes404/ProCode-basic-chatbot.git
cd ProCode-basic-chatbot
```

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install streamlit google-genai python-dotenv pypdf chromadb
```

## API Key Setup

Create a file named `.env` in the project folder:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Never upload `.env` to GitHub.

## Run NOVA

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How RAG Works

1. Upload a PDF in the NOVA sidebar.
2. NOVA extracts the text from the PDF.
3. The text is split into smaller chunks.
4. Gemini creates embeddings for each chunk.
5. ChromaDB stores the chunks and embedBuilt by Kishore.dings locally.
6. When a user asks a question, NOVA retrieves the most relevant document sections.
7. Gemini generates an answer using the retrieved context.
8. NOVA displays the relevant PDF page sources.

## Version History

- **V1** — Basic Gemini chatbot
- **V2** — Chat interface with conversation memory
- **V3** — NOVA RAG Document Chatbot with PDF-based answers

## Author
Kishore M S
