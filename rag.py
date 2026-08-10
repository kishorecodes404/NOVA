import hashlib
from io import BytesIO
from pathlib import Path

import chromadb
from google import genai
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


def index_pdf(uploaded_file, api_key):
    """
    Read an uploaded PDF, split it into chunks,
    embed the chunks, and save them in ChromaDB.
    """
    file_bytes = uploaded_file.getvalue()
    document_id = hashlib.sha256(file_bytes).hexdigest()
    collection = get_collection()

    # Do not add the same PDF twice.
    existing_document = collection.get(
        where={"document_id": document_id}
    )

    if existing_document["ids"]:
        return 0, "This PDF was already added."

    reader = PdfReader(BytesIO(file_bytes))

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        for chunk_number, chunk in enumerate(split_text(page_text), start=1):
            embedding_text = (
                f"title: {uploaded_file.name} | text: {chunk}"
            )

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
        return 0, "No readable text was found in this PDF."

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(documents), "PDF added successfully."


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