import hashlib
from io import BytesIO
from pathlib import Path

import chromadb
from google import genai
from pypdf import PdfReader
import pandas as pd
from docx import Document
from pypdf import PdfReader



CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "nova_documents"
EMBEDDING_MODEL = "gemini-embedding-2"


def get_collection():
    """Create or open NOVA's local vector database."""
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def split_text(text, chunk_size=1000, overlap=150):
    """Split text into small overlapping chunks for better retrieval."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_embedding(text, api_key):
    """Create one Gemini embedding vector."""
    client = genai.Client(api_key=api_key)

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values

def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file."""
    reader = PdfReader(uploaded_file)

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {
                    "text": text,
                    "page": page_number,
                }
            )

    return pages


def extract_text_from_txt(uploaded_file):
    """Extract text from a TXT file."""
    text = uploaded_file.read().decode("utf-8", errors="ignore")

    return [
        {
            "text": text,
            "page": 1,
        }
    ]


def extract_text_from_docx(uploaded_file):
    """Extract text from a DOCX file."""
    document = Document(uploaded_file)

    text = "\n".join(
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    )

    return [
        {
            "text": text,
            "page": 1,
        }
    ]


def extract_text_from_csv(uploaded_file):
    """Extract text from a CSV file."""
    dataframe = pd.read_csv(uploaded_file)
    text = dataframe.to_string(index=False)

    return [
        {
            "text": text,
            "page": 1,
        }
    ]


def extract_text_from_xlsx(uploaded_file):
    """Extract text from an XLSX file."""
    excel_file = pd.ExcelFile(uploaded_file)

    sheets_text = []
    for sheet_name in excel_file.sheet_names:
        dataframe = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        sheet_text = dataframe.to_string(index=False)
        sheets_text.append(f"Sheet: {sheet_name}\n{sheet_text}")

    text = "\n\n".join(sheets_text)

    return [
        {
            "text": text,
            "page": 1,
        }
    ]


def extract_text_from_document(uploaded_file):
    """Extract text from supported document types."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if file_name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)

    if file_name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)

    if file_name.endswith(".csv"):
        return extract_text_from_csv(uploaded_file)

    if file_name.endswith(".xlsx"):
        return extract_text_from_xlsx(uploaded_file)

    if file_name.endswith(".doc"):
        raise ValueError("Old .doc files are not supported. Please convert it to .docx.")

    raise ValueError("Unsupported file type. Please upload PDF, TXT, DOCX, CSV, or XLSX.")

def index_document(uploaded_file, api_key):
    """Index an uploaded document into ChromaDB."""
    file_bytes = uploaded_file.getvalue()
    document_id = hashlib.md5(file_bytes).hexdigest()

    collection = get_collection()

    existing = collection.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )

    if existing["ids"]:
        return 0, "This document was already added."

    pages = extract_text_from_document(uploaded_file)

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for page_data in pages:
        page_number = page_data["page"]
        page_text = page_data["text"]

        for chunk_number, chunk in enumerate(split_text(page_text), start=1):
            embedding_text = f"title: {uploaded_file.name} | text: {chunk}"
            embedding = create_embedding(embedding_text, api_key)

            documents.append(chunk)
            embeddings.append(embedding)
            metadatas.append(
                {
                    "source": uploaded_file.name,
                    "page": page_number,
                    "document_id": document_id,
                }
            )
            ids.append(f"{document_id}-{page_number}-{chunk_number}")

    if not documents:
        return 0, "No readable text was found in this document."

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(documents), "Document added successfully."

def retrieve_context(question, api_key, number_of_results=4):
    """Find the document chunks most relevant to a question."""
    collection = get_collection()

    if collection.count() == 0:
        return "", []

    query_text = f"task: question answering | query: {question}"
    query_embedding = create_embedding(query_text, api_key)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=number_of_results,
        include=["documents", "metadatas"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n---\n\n".join(documents)

    sources = []
    for metadata in metadatas:
        source = f"{metadata['source']} (page {metadata['page']})"

        if source not in sources:
            sources.append(source)

    return context, sources


def clear_documents():
    """Delete all uploaded PDFs and their embeddings."""
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass