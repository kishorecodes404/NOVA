import email
import hashlib
import imaplib
import json
import os
import smtplib
import uuid
from datetime import datetime, timedelta
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import chromadb
import pandas as pd
import requests
import streamlit as st

from docx import Document
from pypdf import PdfReader

try:
    # duckduckgo_search was renamed to ddgs - prefer the maintained
    # package, fall back to the old name if that's what's installed.
    from ddgs import DDGS
    from ddgs.exceptions import RatelimitException
except ImportError:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import RatelimitException

try:
    from icalendar import Calendar
except ImportError:
    Calendar = None


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_SESSION = requests.Session()

CHROMA_PATH = Path("chroma_db")

ADMIN_COLLECTION_NAME = "nova_admin_documents"
USER_COLLECTION_NAME = "nova_user_documents"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def get_chroma_client():
    """Create and cache NOVA's ChromaDB client."""

    return chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )


@st.cache_resource
def get_collection(collection_name):
    """Create or open and cache a NOVA vector collection."""

    chroma_client = get_chroma_client()

    return chroma_client.get_or_create_collection(
        name=collection_name
    )


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(
    text,
    chunk_size=1000,
    overlap=150,
):
    """Split text into overlapping chunks."""

    chunks = []

    start = 0

    step = chunk_size - overlap

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_data(show_spinner=False)
def create_embedding(text):
    """Create a local embedding using Ollama."""

    response = OLLAMA_SESSION.post(
        "http://localhost:11434/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
            "keep_alive": "24h",
            "options": {
                "num_gpu": 99,
            },
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


# ============================================================
# PDF
# ============================================================

def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file."""

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        text = page.extract_text() or ""

        if text.strip():

            pages.append(
                {
                    "text": text,
                    "page": page_number,
                }
            )

    return pages


# ============================================================
# TXT
# ============================================================

def extract_text_from_txt(uploaded_file):
    """Extract text from a TXT file."""

    text = uploaded_file.read().decode(
        "utf-8",
        errors="ignore",
    )

    return [
        {
            "text": text,
            "page": 1,
        }
    ]


# ============================================================
# DOCX
# ============================================================

def extract_text_from_docx(uploaded_file):
    """Extract text from a DOCX file."""

    document = Document(uploaded_file)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [
        {
            "text": text,
            "page": 1,
        }
    ]


# ============================================================
# CSV
# ============================================================

def extract_text_from_csv(uploaded_file):
    """Extract text from a CSV file."""

    dataframe = pd.read_csv(uploaded_file)

    text = dataframe.to_string(
        index=False
    )

    return [
        {
            "text": text,
            "page": 1,
        }
    ]


# ============================================================
# XLSX
# ============================================================

def extract_text_from_xlsx(uploaded_file):
    """Extract text from an XLSX file."""

    excel_file = pd.ExcelFile(
        uploaded_file
    )

    sheets_text = []

    for sheet_name in excel_file.sheet_names:

        dataframe = pd.read_excel(
            uploaded_file,
            sheet_name=sheet_name,
        )

        sheet_text = dataframe.to_string(
            index=False
        )

        sheets_text.append(
            f"Sheet: {sheet_name}\n{sheet_text}"
        )

    text = "\n\n".join(
        sheets_text
    )

    return [
        {
            "text": text,
            "page": 1,
        }
    ]


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_text_from_document(uploaded_file):
    """Extract text from supported document types."""

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(
            uploaded_file
        )

    if file_name.endswith(".txt"):
        return extract_text_from_txt(
            uploaded_file
        )

    if file_name.endswith(".docx"):
        return extract_text_from_docx(
            uploaded_file
        )

    if file_name.endswith(".csv"):
        return extract_text_from_csv(
            uploaded_file
        )

    if file_name.endswith(".xlsx"):
        return extract_text_from_xlsx(
            uploaded_file
        )

    if file_name.endswith(".doc"):
        raise ValueError(
            "Old .doc files are not supported. "
            "Please convert it to .docx."
        )

    raise ValueError(
        "Unsupported file type. "
        "Please upload PDF, TXT, DOCX, CSV, or XLSX."
    )


# ============================================================
# DOCUMENT INDEXING
# ============================================================

def index_document(uploaded_file):
    """Index a user document into ChromaDB."""

    file_bytes = uploaded_file.getvalue()

    document_id = hashlib.md5(
        file_bytes
    ).hexdigest()

    collection = get_collection(
        USER_COLLECTION_NAME
    )

    existing = collection.get(
        where={
            "document_id": document_id
        },
        include=["metadatas"],
    )

    if existing["ids"]:

        return (
            0,
            "This document was already added."
        )

    pages = extract_text_from_document(
        uploaded_file
    )

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for page_data in pages:

        page_number = page_data["page"]
        page_text = page_data["text"]

        chunks = split_text(
            page_text
        )

        for chunk_number, chunk in enumerate(
            chunks,
            start=1,
        ):

            embedding_text = (
                f"title: {uploaded_file.name} | "
                f"text: {chunk}"
            )

            embedding = create_embedding(
                embedding_text
            )

            documents.append(chunk)

            embeddings.append(
                embedding
            )

            metadatas.append(
                {
                    "source": uploaded_file.name,
                    "page": page_number,
                    "document_id": document_id,
                }
            )

            ids.append(
                f"{document_id}-"
                f"{page_number}-"
                f"{chunk_number}"
            )

    if not documents:

        return (
            0,
            "No readable text was found in this document."
        )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return (
        len(documents),
        "Document added successfully."
    )


# ============================================================
# ADMIN DOCUMENT INDEXING
# ============================================================

def index_admin_document(uploaded_file):
    """Index a document into NOVA's permanent admin knowledge base."""

    file_bytes = uploaded_file.getvalue()

    document_id = hashlib.md5(
        file_bytes
    ).hexdigest()

    collection = get_collection(
        ADMIN_COLLECTION_NAME
    )

    existing = collection.get(
        where={
            "document_id": document_id
        },
        include=["metadatas"],
    )

    if existing["ids"]:

        return (
            0,
            "This admin document was already added."
        )

    pages = extract_text_from_document(
        uploaded_file
    )

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for page_data in pages:

        page_number = page_data["page"]
        page_text = page_data["text"]

        chunks = split_text(
            page_text
        )

        for chunk_number, chunk in enumerate(
            chunks,
            start=1,
        ):

            embedding_text = (
                f"title: {uploaded_file.name} | "
                f"text: {chunk}"
            )

            embedding = create_embedding(
                embedding_text
            )

            documents.append(chunk)

            embeddings.append(
                embedding
            )

            metadatas.append(
                {
                    "source": uploaded_file.name,
                    "page": page_number,
                    "document_id": document_id,
                    "type": "admin",
                }
            )

            ids.append(
                f"admin-{document_id}-"
                f"{page_number}-"
                f"{chunk_number}"
            )

    if not documents:

        return (
            0,
            "No readable text was found in this document."
        )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return (
        len(documents),
        "Admin document added successfully."
    )


# ============================================================
# LIST INDEXED DOCUMENTS
#
# Used by the sidebar to show what's actually in the knowledge
# base right now - separate from index_admin_document()/
# index_document() above, which only ADD to it. Reads both
# collections and dedupes by document_id (each source document is
# stored as many chunks - one row per chunk, not per document).
# ============================================================

def list_indexed_documents():
    """
    Returns a list of every distinct document currently indexed,
    across both the admin (permanent) and user collections.

    Returns:
        list of dicts, each: {
            "source": original filename,
            "chunk_count": number of chunks stored for it,
            "collection": "admin" or "user",
        }
        Sorted by source name.
    """

    documents_by_id = {}

    for collection_name, collection_label in (
        (ADMIN_COLLECTION_NAME, "admin"),
        (USER_COLLECTION_NAME, "user"),
    ):

        try:
            collection = get_collection(collection_name)
            existing = collection.get(include=["metadatas"])
        except Exception as error:
            print(
                f"[LIST DOCUMENTS] Couldn't read {collection_name}: {error}",
                flush=True,
            )
            continue

        for metadata in existing.get("metadatas", []) or []:

            if not metadata:
                continue

            document_id = metadata.get("document_id")
            source = metadata.get("source", "Unknown")

            key = (collection_label, document_id or source)

            if key not in documents_by_id:
                documents_by_id[key] = {
                    "source": source,
                    "chunk_count": 0,
                    "collection": collection_label,
                }

            documents_by_id[key]["chunk_count"] += 1

    return sorted(
        documents_by_id.values(),
        key=lambda entry: entry["source"].lower(),
    )


# ============================================================
# CONTEXT RETRIEVAL
# ============================================================

def retrieve_context(question, number_of_results=4):
    """Retrieve relevant document context efficiently."""

    admin_collection = get_collection(ADMIN_COLLECTION_NAME)
    user_collection = get_collection(USER_COLLECTION_NAME)

    # ------------------------------------------------
    # IMPORTANT:
    # If there are no documents at all, do NOT call
    # the embedding model.
    # ------------------------------------------------

    admin_count = admin_collection.count()
    user_count = user_collection.count()

    if admin_count == 0 and user_count == 0:
        return "", []

    # ------------------------------------------------
    # Only create an embedding when documents exist.
    # ------------------------------------------------

    query_embedding = create_embedding(question)

    all_results = []

    collections = [
        (admin_collection, admin_count, "admin"),
        (user_collection, user_count, "user"),
    ]

    for collection, collection_count, collection_type in collections:

        if collection_count == 0:
            continue

        results_to_get = min(
            number_of_results,
            collection_count,
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=results_to_get,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get("documents")

        if not documents or not documents[0]:
            continue

        for i, document in enumerate(documents[0]):

            all_results.append(
                {
                    "document": document,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "type": collection_type,
                }
            )

    if not all_results:
        return "", []

    # Best similarity first
    all_results.sort(
        key=lambda item: item["distance"]
    )

    selected_results = all_results[:number_of_results]

    context = "\n\n---\n\n".join(
        item["document"]
        for item in selected_results
    )

    sources = []

    for item in selected_results:

        metadata = item["metadata"]

        source = (
            f"{metadata['source']} "
            f"(page {metadata['page']})"
        )

        if source not in sources:
            sources.append(source)

    return context, sources

# ============================================================
# WEB SEARCH (WEB AGENT)
# ============================================================

# ============================================================
# WEB SEARCH (WEB AGENT)
# ============================================================

def _infer_timelimit(query):
    """
    Guess how fresh the results need to be, based on the question.

    'd' (past day)  - today / right now / live / breaking
    'w' (past week) - latest / current / recent / price / stock / news
    None            - no recency signal, let DDG rank normally
    """

    q = query.lower()

    day_signals = ("today", "right now", "currently", "live", "breaking")
    if any(signal in q for signal in day_signals):
        return "d"

    week_signals = (
        "current", "latest", "recent", "recently", "this week",
        "price", "prices", "stock", "stocks", "score", "news",
    )
    if any(signal in q for signal in week_signals):
        return "w"

    return None


def web_search(query, max_results=6):
    """
    Search the web using DuckDuckGo.

    Returns:
        context: formatted search result text
        sources: result URLs
    """

    import time

    search_start = time.perf_counter()

    timelimit = _infer_timelimit(query)

    results = []
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):

        try:

            with DDGS() as ddgs:

                search_results = ddgs.text(
                    query,
                    max_results=max_results,
                    timelimit=timelimit,
                )

                results = list(search_results) if search_results else []

                # A narrow freshness window can legitimately return
                # nothing (e.g. "current" phrasing on a topic that
                # isn't actually breaking news). Don't let that turn
                # into "no answer" - fall back to an unfiltered
                # search rather than returning empty-handed.
                if not results and timelimit:

                    print(
                        f"[WEB SEARCH] no results within timelimit="
                        f"{timelimit!r}, retrying unfiltered.",
                        flush=True,
                    )

                    search_results = ddgs.text(
                        query,
                        max_results=max_results,
                    )

                    results = list(search_results) if search_results else []

            break

        except RatelimitException:

            # DuckDuckGo rate-limits automated/repeated queries
            # (HTTP 202) - this is common and transient, not a real
            # failure. Back off and retry a couple of times before
            # giving up, rather than immediately telling the user
            # "no information found" for what's actually a
            # throttling issue on our end.
            if attempt == max_attempts:

                print(
                    f"[WEB SEARCH] rate-limited after "
                    f"{max_attempts} attempts, giving up.",
                    flush=True,
                )

                return "", []

            wait_seconds = attempt  # 1s, then 2s

            print(
                f"[WEB SEARCH] rate-limited, retrying in "
                f"{wait_seconds}s (attempt {attempt}/{max_attempts})",
                flush=True,
            )

            time.sleep(wait_seconds)

        except Exception as error:

            print(
                f"[WEB SEARCH ERROR] {error}",
                flush=True,
            )

            return "", []

    if not results:

        print(
            "[WEB SEARCH] No results returned.",
            flush=True,
        )

        return "", []

    chunks = []
    sources = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        if not isinstance(result, dict):
            continue

        title = str(
            result.get("title") or ""
        ).strip()

        body = str(
            result.get("body") or ""
        ).strip()

        href = str(
            result.get("href") or ""
        ).strip()

        if not title and not body:
            continue

        chunks.append(
            f"""SOURCE {index}
TITLE: {title}
CONTENT:
{body}"""
        )

        if href and href not in sources:
            sources.append(href)

    if not chunks:

        print(
            "[WEB SEARCH] No usable results.",
            flush=True,
        )

        return "", []

    context = "\n\n====================\n\n".join(
        chunks
    )

    print(
        f"[WEB SEARCH] "
        f"{len(chunks)} usable results | "
        f"{len(sources)} sources | "
        f"{time.perf_counter() - search_start:.2f}s",
        flush=True,
    )

    print(
        "[WEB SEARCH EVIDENCE]\n"
        + context,
        flush=True,
    )

    return context, sources
# ============================================================
# CLEAR USER DOCUMENTS
# ============================================================

def clear_documents():
    """
    Delete all temporary user documents
    while keeping the collection itself.
    """

    collection = get_collection(
        USER_COLLECTION_NAME
    )

    existing = collection.get()

    if existing["ids"]:

        collection.delete(
            ids=existing["ids"]
        )


# ============================================================
# MAIL AGENT
#
# Searches a mailbox over IMAP. Configure via environment
# variables (works with Gmail app passwords, Outlook, or any
# standard IMAP provider):
#
#     NOVA_IMAP_HOST      e.g. "imap.gmail.com"
#     NOVA_IMAP_USER      e.g. "you@gmail.com"
#     NOVA_IMAP_PASSWORD  an app password, not your normal login
#     NOVA_IMAP_FOLDER    defaults to "INBOX"
#
# If unconfigured, search_mail() returns no evidence rather than
# raising - the MAIL prompt already handles "no evidence found" by
# telling the user honestly rather than guessing.
# ============================================================

IMAP_HOST = os.environ.get("NOVA_IMAP_HOST", "")
IMAP_USER = os.environ.get("NOVA_IMAP_USER", "")
IMAP_PASSWORD = os.environ.get("NOVA_IMAP_PASSWORD", "")
IMAP_FOLDER = os.environ.get("NOVA_IMAP_FOLDER", "INBOX")


def _decode_mime_words(value):

    if not value:
        return ""

    decoded = ""

    for part, encoding in decode_header(value):

        if isinstance(part, bytes):
            decoded += part.decode(encoding or "utf-8", errors="ignore")
        else:
            decoded += part

    return decoded


def _extract_plain_text(msg):

    if msg.is_multipart():

        for part in msg.walk():

            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition") or "")

            if content_type == "text/plain" and "attachment" not in disposition:

                try:
                    return part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8",
                        errors="ignore",
                    )
                except Exception:
                    continue

        return ""

    try:
        return msg.get_payload(decode=True).decode(
            msg.get_content_charset() or "utf-8",
            errors="ignore",
        )
    except Exception:
        return ""


def search_mail(query, max_results=5, scan_limit=50, require_keyword_match=False):
    """
    Search the configured mailbox for messages relevant to `query`.

    Efficiency note: this does ONE search + ONE batched header
    fetch across up to `scan_limit` recent messages (single round
    trip), scores them by keyword overlap, and only fetches the
    full body for the top `max_results` candidates. It does not
    fetch every message in full - that would be slow against any
    mailbox with real volume.

    Args:
        require_keyword_match: if True, skip the "no keyword ->
            fall back to most recent messages" behavior below and
            return no evidence instead. That fallback exists for
            read-style questions with no distinguishing keyword
            ("summarize my last email") - it should NEVER be used
            to answer "does an email from X exist", since it would
            then return unrelated recent mail as if it matched X.
            Set this True for any caller that treats the result as
            confirmation of a specific match (e.g. resolving a
            person's name to their email address before sending).

    Returns:
        context: formatted text block for the LLM prompt
        sources: list of short "SUBJECT - FROM" labels for the UI
    """

    if not (IMAP_HOST and IMAP_USER and IMAP_PASSWORD):
        print("[MAIL SEARCH] Not configured - skipping.", flush=True)
        return "", []

    import time
    search_start = time.perf_counter()

    connection = None

    try:
        connection = imaplib.IMAP4_SSL(IMAP_HOST)
        connection.login(IMAP_USER, IMAP_PASSWORD)
        connection.select(IMAP_FOLDER, readonly=True)

        status, data = connection.search(None, "ALL")

        if status != "OK" or not data or not data[0]:
            return "", []

        scan_ids = data[0].split()[-scan_limit:]

        if not scan_ids:
            return "", []

        id_set = b",".join(scan_ids)

        status, header_data = connection.fetch(
            id_set,
            "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])",
        )

        if status != "OK" or not header_data:
            return "", []

        mail_stopwords = {
            "the", "and", "for", "are", "was", "were", "did", "does",
            "what", "who", "when", "where", "which", "how", "from",
            "about", "any", "that", "this", "has", "have", "did",
            "mail", "mails", "email", "emails", "e-mail", "e-mails",
            "inbox", "message", "messages", "summarize", "summarise",
            "show", "tell", "give", "find", "get", "got", "receive",
            "received", "recent", "recently", "last", "latest",
            "new", "unread", "read", "please", "can", "you",
        }

        keywords = [
            token.strip("?.,!'\"")
            for token in query.lower().split()
            if len(token.strip("?.,!'\"")) > 2
            and token.strip("?.,!'\"") not in mail_stopwords
        ]

        candidates = []
        id_index = 0

        for item in header_data:

            if not isinstance(item, tuple):
                continue

            if id_index >= len(scan_ids):
                break

            msg_id = scan_ids[id_index]
            id_index += 1

            header_msg = email.message_from_bytes(item[1])

            subject = _decode_mime_words(header_msg.get("Subject", ""))
            sender = _decode_mime_words(header_msg.get("From", ""))
            date = header_msg.get("Date", "")

            haystack = f"{subject} {sender}".lower()
            score = sum(1 for kw in keywords if kw in haystack)

            candidates.append((score, msg_id, subject, sender, date))

        keyword_matches = [c for c in candidates if c[0] > 0]

        if keyword_matches:
            # A specific keyword search ("emails from Sarah about
            # the budget") - rank by relevance.
            keyword_matches.sort(key=lambda c: c[0], reverse=True)
            top_candidates = keyword_matches[:max_results]
        elif require_keyword_match:
            # Caller needs a genuine match (e.g. resolving a name to
            # an address) - returning unrelated recent mail here
            # would look like a match and isn't one. Fail closed.
            print(
                f"[MAIL SEARCH] No keyword match for '{query}' - "
                "require_keyword_match set, returning no evidence.",
                flush=True,
            )
            return "", []
        else:
            # No distinguishing keyword ("summarize my last email",
            # "what's my most recent email") - there's nothing to
            # rank by relevance, so fall back to the most recent
            # messages instead of returning nothing. scan_ids is
            # oldest-first, so reverse for newest-first.
            top_candidates = list(reversed(candidates))[:max_results]

        chunks = []
        sources = []

        for _, msg_id, subject, sender, date in top_candidates:

            status, msg_data = connection.fetch(msg_id, "(RFC822)")

            if status != "OK" or not msg_data or not msg_data[0]:
                continue

            msg = email.message_from_bytes(msg_data[0][1])
            body = _extract_plain_text(msg)[:800]

            chunks.append(
                f"""EMAIL {len(chunks) + 1}
FROM: {sender}
SUBJECT: {subject}
DATE: {date}
BODY:
{body}"""
            )
            sources.append(f"{subject} - {sender}")

        context = "\n\n====================\n\n".join(chunks)

        print(
            f"[MAIL SEARCH] {len(chunks)} usable results | "
            f"{time.perf_counter() - search_start:.2f}s",
            flush=True,
        )

        return context, sources

    except Exception as error:

        print(f"[MAIL SEARCH ERROR] {error}", flush=True)
        return "", []

    finally:

        if connection is not None:
            try:
                connection.logout()
            except Exception:
                pass


# ============================================================
# MAIL AGENT - SEND
#
# Sends over SMTP. Configure via environment variables (works with
# Gmail app passwords, Outlook, or any standard SMTP provider):
#
#     NOVA_SMTP_HOST      e.g. "smtp.gmail.com"
#     NOVA_SMTP_PORT      defaults to 587 (STARTTLS)
#     NOVA_SMTP_USER      defaults to NOVA_IMAP_USER if unset
#     NOVA_SMTP_PASSWORD  defaults to NOVA_IMAP_PASSWORD if unset
#
# Most providers use the same login for IMAP and SMTP, so the SMTP
# vars are optional and fall back to the IMAP ones - but they're
# separate on purpose in case a provider (or a "send from a
# different address" setup) needs different credentials.
#
# Unlike search_mail(), a missing config here is a hard failure,
# not a silent skip - the caller asked NOVA to send something, and
# silently doing nothing would be worse than telling them why it
# didn't happen.
# ============================================================

SMTP_HOST = os.environ.get("NOVA_SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("NOVA_SMTP_PORT", "587") or "587")
SMTP_USER = os.environ.get("NOVA_SMTP_USER", "") or IMAP_USER
SMTP_PASSWORD = os.environ.get("NOVA_SMTP_PASSWORD", "") or IMAP_PASSWORD


def _split_addresses(value):

    if not value:
        return []

    if isinstance(value, str):
        return [addr.strip() for addr in value.split(",") if addr.strip()]

    return [addr.strip() for addr in value if str(addr).strip()]


def send_mail(to, subject, body, cc=None, bcc=None):
    """
    Sends a plain-text email over SMTP.

    Args:
        to: recipient address(es) - comma-separated string or list
        subject: subject line
        body: plain-text body
        cc: optional cc address(es) - comma-separated string or list
        bcc: optional bcc address(es) - comma-separated string or
            list. Deliberately never written to a message header
            (that's the whole point of bcc) - it's only added to
            the SMTP envelope recipient list below, so To/Cc
            recipients never see it.

    Returns:
        (success: bool, message: str) - message is a short,
        user-facing description of what happened (or why it
        didn't), meant to be shown directly in the UI.
    """

    if not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD):
        return False, (
            "Sending isn't configured - set NOVA_SMTP_HOST, "
            "NOVA_SMTP_USER and NOVA_SMTP_PASSWORD (an app "
            "password, not your normal login)."
        )

    to_list = _split_addresses(to)
    cc_list = _split_addresses(cc)
    bcc_list = _split_addresses(bcc)

    if not to_list:
        return False, "No recipient address was given."

    message = MIMEMultipart()
    message["From"] = SMTP_USER
    message["To"] = ", ".join(to_list)

    if cc_list:
        message["Cc"] = ", ".join(cc_list)

    # No "Bcc" header is added to the message itself - bcc only
    # goes into the SMTP envelope recipients below, which is what
    # keeps it hidden from the To/Cc recipients.

    message["Subject"] = subject or "(no subject)"
    message.attach(MIMEText(body or "", "plain"))

    all_recipients = to_list + cc_list + bcc_list

    try:
        # Port 465 is implicit TLS (SMTP_SSL) - calling starttls() on
        # that port fails with a TLS handshake error, since the
        # connection is already expected to be encrypted from the
        # start. Port 587 (and everything else) is plaintext-then-
        # upgrade, so starttls() is what's needed there instead.
        if SMTP_PORT == 465:
            smtp_class = smtplib.SMTP_SSL
        else:
            smtp_class = smtplib.SMTP

        with smtp_class(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            if SMTP_PORT != 465:
                server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, all_recipients, message.as_string())

        print(
            f"[MAIL SEND] Sent to {to_list} (cc: {cc_list}, bcc: {bcc_list})",
            flush=True,
        )

        return True, f"Email sent to {', '.join(to_list)}."

    except Exception as error:

        print(f"[MAIL SEND ERROR] {error}", flush=True)
        return False, f"Couldn't send that email: {error}"


# ============================================================
# MEETINGS AGENT
#
# Reads (and now writes) a local .ics calendar file. Configure via:
#
#     NOVA_MEETINGS_ICS_PATH   path to a .ics file
#
# This is deliberately the simplest possible backend - no OAuth,
# no external API. Swap this out for the Google Calendar API or
# CalDAV later without touching app.py: search_meetings() and
# schedule_meeting() just need to keep their same signatures.
#
# IMPORTANT CAVEAT for schedule_meeting(): this only writes to the
# local .ics file at NOVA_MEETINGS_ICS_PATH. If that file is just a
# read-only export or a subscription copy of a calendar hosted
# elsewhere (Google Calendar, Outlook, etc), writing to it will NOT
# push the event back to that real calendar, and attendees will NOT
# get an invite - the "attendee" field is stored on the event for
# your own record only. For real scheduling with invites sent to
# attendees, this needs to be swapped out for the Google Calendar
# API or CalDAV, per the note above. If NOVA_MEETINGS_ICS_PATH
# points at a file your own calendar app actively watches (e.g. a
# local .ics your calendar app is subscribed to by file path), new
# events will show up there once it re-reads the file.
#
# Requires the `icalendar` package (pip install icalendar).
# ============================================================

MEETINGS_ICS_PATH = os.environ.get("NOVA_MEETINGS_ICS_PATH", "")

# ============================================================
# MULTI-USER CALENDARS
#
# NOVA_MEETINGS_ICS_PATH stays the primary/default user's calendar
# (backward compatible with single-user setups - key "me").
#
# For checking OTHER people's schedules too (e.g. "is Priya free
# Thursday?" or catching a scheduling conflict on an attendee's own
# calendar before booking a meeting), configure additional users
# via:
#
#     NOVA_MEETINGS_ICS_PATHS="priya:/path/priya.ics,dev:/path/dev-team.ics"
#
# - comma-separated "name:path" pairs. Names are matched
# case-insensitively wherever a user is looked up (by name, or by
# the local part of an email address).
# ============================================================


def _parse_user_calendar_map():
    """Builds {user_name_lower: ics_path} from the env vars above."""

    mapping = {}

    if MEETINGS_ICS_PATH:
        mapping["me"] = MEETINGS_ICS_PATH

    extra = os.environ.get("NOVA_MEETINGS_ICS_PATHS", "")

    for entry in extra.split(","):

        entry = entry.strip()

        if not entry or ":" not in entry:
            continue

        name, path = entry.split(":", 1)
        name = name.strip().lower()
        path = path.strip()

        if name and path:
            mapping[name] = path

    return mapping


MEETINGS_CALENDARS = _parse_user_calendar_map()


def get_configured_calendar_users():
    """Names of every user with a configured calendar (lowercased)."""

    return list(MEETINGS_CALENDARS.keys())


def resolve_calendar_user(name_or_email):
    """
    Matches a name or email against configured calendar users.

    Tries, in order: exact match, the email's local part (before
    '@'), then a substring match either way (so "Priya" matches a
    configured "priya sharma", and vice versa).

    Returns the matched key into MEETINGS_CALENDARS, or None.
    """

    if not name_or_email:
        return None

    candidate = name_or_email.strip().lower()

    if "@" in candidate:
        candidate = candidate.split("@", 1)[0]

    if candidate in MEETINGS_CALENDARS:
        return candidate

    for user_name in MEETINGS_CALENDARS:
        if user_name in candidate or candidate in user_name:
            return user_name

    return None


def _read_ics_calendar(path):
    """Loads and parses an .ics file, or None if it can't be read."""

    if not path:
        return None

    try:
        with open(path, "rb") as ics_file:
            return Calendar.from_ical(ics_file.read())
    except Exception as error:
        print(f"[MEETINGS] Couldn't read calendar at {path}: {error}", flush=True)
        return None


def search_meetings(query, max_results=5, days_behind=7, days_ahead=30, user=None):
    """
    Search a configured calendar for events relevant to `query`,
    within a window from `days_behind` days ago to `days_ahead`
    days from now.

    Args:
        user: which configured calendar to search - a key from
            MEETINGS_CALENDARS (e.g. "priya"), matched via
            resolve_calendar_user(). Defaults to the primary user
            ("me" / NOVA_MEETINGS_ICS_PATH) for backward
            compatibility with single-user setups.

    Returns:
        context: formatted text block for the LLM prompt
        sources: list of short "TITLE (date)" labels for the UI
    """

    calendar_user = user or "me"
    ics_path = MEETINGS_CALENDARS.get(calendar_user, MEETINGS_ICS_PATH)

    if not ics_path:
        print("[MEETINGS SEARCH] Not configured - skipping.", flush=True)
        return "", []

    if Calendar is None:
        print(
            "[MEETINGS SEARCH] 'icalendar' package not installed - "
            "run `pip install icalendar`.",
            flush=True,
        )
        return "", []

    import time
    search_start = time.perf_counter()

    calendar = _read_ics_calendar(ics_path)

    if calendar is None:
        return "", []

    now = datetime.now()
    window_start = now - timedelta(days=days_behind)
    window_end = now + timedelta(days=days_ahead)

    keywords = [
        token.strip("?.,!'\"")
        for token in query.lower().split()
        if len(token.strip("?.,!'\"")) > 2
    ]

    events = []

    for component in calendar.walk():

        if component.name != "VEVENT":
            continue

        try:
            summary = str(component.get("summary", "") or "")
            location = str(component.get("location", "") or "")
            description = str(component.get("description", "") or "")

            dtstart_field = component.get("dtstart")
            dtstart_val = dtstart_field.dt if dtstart_field else None

            if dtstart_val is None:
                continue

            if isinstance(dtstart_val, datetime):
                # dtstart_val is often UTC-aware (most calendar
                # exports - Google/Outlook - write DTSTART in UTC
                # with a Z suffix). window_start/window_end are
                # built from datetime.now(), which is naive LOCAL
                # time. Stripping tzinfo without converting first
                # kept the UTC clock numbers as if they were local,
                # shifting every event by the UTC offset - enough
                # to push events near midnight into the wrong day
                # and drop them from the search window entirely.
                # astimezone() with no args converts to the system
                # local timezone first, THEN we drop the label.
                if dtstart_val.tzinfo is not None:
                    start_naive = dtstart_val.astimezone().replace(tzinfo=None)
                else:
                    start_naive = dtstart_val
            else:
                start_naive = datetime(
                    dtstart_val.year, dtstart_val.month, dtstart_val.day
                )

            if not (window_start <= start_naive <= window_end):
                continue

            haystack = f"{summary} {location} {description}".lower()
            score = sum(1 for kw in keywords if kw in haystack)

            # Unlike mail, a meeting is still a candidate with zero
            # keyword overlap - "what's on my calendar tomorrow"
            # has no topic keyword, it's purely date-based. Sort by
            # relevance first, then chronologically.
            events.append(
                (score, start_naive, summary, location, description)
            )

        except Exception:
            continue

    if not events:
        return "", []

    events.sort(key=lambda e: (-e[0], e[1]))
    top_events = events[:max_results]

    # Only label events with whose calendar they came from once
    # more than one user is actually configured - a single-user
    # setup shouldn't clutter every event with "(me)".
    owner_label = f" ({calendar_user})" if len(MEETINGS_CALENDARS) > 1 else ""

    chunks = []
    sources = []

    for _, start_naive, summary, location, description in top_events:

        chunks.append(
            f"""EVENT {len(chunks) + 1}{(' - CALENDAR: ' + calendar_user) if owner_label else ''}
TITLE: {summary}
WHEN: {start_naive.strftime('%Y-%m-%d %H:%M')}
LOCATION: {location or 'N/A'}
DETAILS:
{description[:400]}"""
        )
        sources.append(
            f"{summary} ({start_naive.strftime('%Y-%m-%d %H:%M')}){owner_label}"
        )

    context = "\n\n====================\n\n".join(chunks)

    print(
        f"[MEETINGS SEARCH] {len(chunks)} usable results for '{calendar_user}' | "
        f"{time.perf_counter() - search_start:.2f}s",
        flush=True,
    )

    return context, sources


def search_meetings_multi(query, users, max_results=5, days_behind=7, days_ahead=30):
    """
    Runs search_meetings() across several configured users' calendars
    and merges the results - used for "what's on the calendar" style
    questions that name more than one person (or "the team").

    Args:
        users: list of configured calendar user keys (see
            resolve_calendar_user()). Unknown/unconfigured names are
            silently skipped.

    Returns:
        context: combined formatted text block for the LLM prompt
        sources: combined list of "TITLE (date) (user)" labels
    """

    all_context = []
    all_sources = []

    for user in users:

        if user not in MEETINGS_CALENDARS:
            continue

        context, sources = search_meetings(
            query,
            max_results=max_results,
            days_behind=days_behind,
            days_ahead=days_ahead,
            user=user,
        )

        if context:
            all_context.append(context)
        all_sources.extend(sources)

    return "\n\n====================\n\n".join(all_context), all_sources


def check_group_availability(attendees, start, end):
    """
    Checks each attendee's configured calendar for events that
    overlap the proposed [start, end) meeting window - a "does this
    time actually work for everyone" pass, run before a meeting
    with named attendees is actually scheduled.

    Args:
        attendees: comma-separated string or list of names/emails
            (as given by the user - e.g. "Priya, bob@company.com").
        start, end: proposed meeting window (datetime).

    Returns:
        conflicts: {matched_user_name: [conflicting event summary
            strings]} - only users with an actual overlap appear.
        unchecked: list of attendee names/emails that don't match
            any configured calendar, so couldn't be checked at all.
    """

    conflicts = {}
    unchecked = []

    if Calendar is None or start is None or end is None:
        return conflicts, [
            a for a in _split_addresses(attendees)
        ]

    for attendee in _split_addresses(attendees):

        matched_user = resolve_calendar_user(attendee)

        if not matched_user:
            unchecked.append(attendee)
            continue

        calendar = _read_ics_calendar(MEETINGS_CALENDARS[matched_user])

        if calendar is None:
            unchecked.append(attendee)
            continue

        overlaps = []

        for component in calendar.walk():

            if component.name != "VEVENT":
                continue

            try:
                summary = str(component.get("summary", "") or "")

                dtstart_field = component.get("dtstart")
                dtend_field = component.get("dtend")

                if not dtstart_field:
                    continue

                event_start = dtstart_field.dt
                event_end = dtend_field.dt if dtend_field else event_start

                # Same UTC-stripped-as-local bug as in
                # search_meetings() above - convert to local time
                # via astimezone() before dropping tzinfo, or the
                # overlap test below compares against `start`/`end`
                # (local, from the user's request) using event times
                # that are silently still in UTC, causing real
                # conflicts to be missed near timezone-offset
                # boundaries.
                if isinstance(event_start, datetime):
                    if event_start.tzinfo is not None:
                        event_start = event_start.astimezone().replace(tzinfo=None)
                else:
                    event_start = datetime(
                        event_start.year, event_start.month, event_start.day
                    )

                if isinstance(event_end, datetime):
                    if event_end.tzinfo is not None:
                        event_end = event_end.astimezone().replace(tzinfo=None)
                else:
                    event_end = datetime(
                        event_end.year, event_end.month, event_end.day
                    )

                # Standard interval overlap test.
                if event_start < end and event_end > start:
                    overlaps.append(
                        f"{summary} ({event_start.strftime('%Y-%m-%d %H:%M')}"
                        f"-{event_end.strftime('%H:%M')})"
                    )

            except Exception:
                continue

        if overlaps:
            conflicts[matched_user] = overlaps

    return conflicts, unchecked


def schedule_meeting(
    title, start, end=None, location="", description="", attendees=None
):
    """
    Adds a new VEVENT to the configured local .ics calendar file.

    Args:
        title: event summary/title
        start: datetime for the event start
        end: datetime for the event end (defaults to start + 30 min)
        location: optional location string
        description: optional description/notes
        attendees: optional attendee address(es) - comma-separated
            string or list. Stored on the event for reference; see
            the module note above about this NOT sending real
            invites unless swapped out for a real calendar API.

    Returns:
        (success: bool, message: str) - message is a short,
        user-facing description meant to be shown directly in the
        UI.
    """

    if not MEETINGS_ICS_PATH:
        return False, (
            "Scheduling isn't configured - set "
            "NOVA_MEETINGS_ICS_PATH to a .ics file."
        )

    if Calendar is None:
        return False, (
            "'icalendar' package not installed - "
            "run `pip install icalendar`."
        )

    from icalendar import Event

    if start is None:
        return False, "No start time was given for that event."

    if end is None:
        end = start + timedelta(minutes=30)

    attendee_list = _split_addresses(attendees)

    try:
        if os.path.exists(MEETINGS_ICS_PATH):
            with open(MEETINGS_ICS_PATH, "rb") as ics_file:
                calendar = Calendar.from_ical(ics_file.read())
        else:
            calendar = Calendar()
            calendar.add("prodid", "-//NOVA//Meetings Agent//EN")
            calendar.add("version", "2.0")
    except Exception as error:
        print(f"[MEETINGS SCHEDULE ERROR] read: {error}", flush=True)
        return False, f"Couldn't read the calendar file: {error}"

    event = Event()
    event.add("summary", title or "Meeting")
    event.add("dtstart", start)
    event.add("dtend", end)
    event.add("dtstamp", datetime.now())
    event.add("uid", f"{uuid.uuid4()}@nova")

    if location:
        event.add("location", location)

    if description:
        event.add("description", description)

    for address in attendee_list:
        event.add("attendee", f"MAILTO:{address}")

    calendar.add_component(event)

    try:
        with open(MEETINGS_ICS_PATH, "wb") as ics_file:
            ics_file.write(calendar.to_ical())
    except Exception as error:
        print(f"[MEETINGS SCHEDULE ERROR] write: {error}", flush=True)
        return False, f"Couldn't write to the calendar file: {error}"

    when_text = start.strftime("%Y-%m-%d %H:%M")

    print(
        f"[MEETINGS SCHEDULE] Added '{title}' at {when_text}",
        flush=True,
    )

    return True, f"Added '{title}' on {when_text} to the calendar."


# ============================================================
# LEAVE AGENT
#
# Applies for leave with a full set of pre-submission validations:
#   - date range validity (end >= start, not absurdly long)
#   - past dates (can't request leave that already happened)
#   - eligibility (is this a recognized user)
#   - weekends/holidays (don't burn balance on non-working days)
#   - leave balance (enough days left of that leave type)
#   - duplicate/overlapping requests (already has leave that period)
#   - existing calendar events during the requested period
#
# Storage is a single local JSON file (no external HR system) -
# configure via NOVA_LEAVE_STORE_PATH. Swap this out for a real HRIS
# API later without touching app.py: validate_leave_request() and
# apply_leave() just need to keep their same signatures.
#
#     NOVA_LEAVE_STORE_PATH        path to a JSON file (default
#                                   "leave_store.json")
#     NOVA_LEAVE_ELIGIBLE_USERS    comma-separated list of user
#                                   names eligible for leave. If
#                                   unset, everyone is eligible
#                                   (permissive default).
#     NOVA_LEAVE_HOLIDAYS          comma-separated YYYY-MM-DD dates
#                                   treated as company holidays.
#     NOVA_LEAVE_MAX_SPAN_DAYS     longest single request allowed,
#                                   inclusive (default 30).
#     NOVA_LEAVE_USER_NAME         display name shown in the
#                                   leave-approval email in place of
#                                   the internal "me" user key
#                                   (default: "Team Member").
# ============================================================

LEAVE_STORE_PATH = os.environ.get("NOVA_LEAVE_STORE_PATH", "leave_store.json")

# Starting/annual allocation per leave type for a user who has never
# been seen before - only used the first time a balance is looked up
# or a request is applied for that user.
DEFAULT_LEAVE_BALANCES = {"annual": 18, "sick": 10, "casual": 6}

LEAVE_MAX_SPAN_DAYS = int(os.environ.get("NOVA_LEAVE_MAX_SPAN_DAYS", "30") or "30")

# The person who approves leave (a manager, HR, etc.) - the request
# email goes here. Reuses the same NOVA_SMTP_* account configured
# for the Mail Agent as the SENDER (see send_mail() above), so the
# approver receives it as coming from the logged-in user's own
# mailbox - exactly like a person forwarding a leave request by
# hand, just automated.
LEAVE_APPROVER_EMAIL = os.environ.get("NOVA_LEAVE_APPROVER_EMAIL", "")

# Display name for the requester shown in the leave-approval email
# (subject + body). NOVA is single-tenant, so the requester is
# always the literal user key "me" internally - but "me has
# requested leave" reads wrong to an approver. Configure the actual
# person's name here; falls back to a neutral placeholder (never the
# literal "me") if unset.
LEAVE_USER_DISPLAY_NAME = os.environ.get("NOVA_LEAVE_USER_NAME", "").strip()


def _parse_holiday_set():
    """Builds a set of date objects from NOVA_LEAVE_HOLIDAYS."""

    holidays = set()

    for part in os.environ.get("NOVA_LEAVE_HOLIDAYS", "").split(","):
        part = part.strip()

        if not part:
            continue

        try:
            holidays.add(datetime.strptime(part, "%Y-%m-%d").date())
        except ValueError:
            continue

    return holidays


LEAVE_HOLIDAYS = _parse_holiday_set()


def is_weekend(day):
    """True if `day` (a date) falls on Saturday or Sunday."""

    return day.weekday() >= 5


def is_holiday(day):
    """True if `day` (a date) is a configured company holiday."""

    return day in LEAVE_HOLIDAYS


def _daterange(start_date, end_date):
    """Yields every date from start_date to end_date, inclusive."""

    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _load_leave_store():
    """
    Loads the JSON leave store, or an empty-but-valid shape if the
    file doesn't exist yet or can't be read.
    """

    if not os.path.exists(LEAVE_STORE_PATH):
        return {"balances": {}, "requests": []}

    try:
        with open(LEAVE_STORE_PATH, "r", encoding="utf-8") as store_file:
            data = json.load(store_file)
    except Exception as error:
        print(f"[LEAVE] Couldn't read store at {LEAVE_STORE_PATH}: {error}", flush=True)
        return {"balances": {}, "requests": []}

    data.setdefault("balances", {})
    data.setdefault("requests", [])
    return data


def _save_leave_store(store):
    """Writes the leave store back to disk. Returns True on success."""

    try:
        with open(LEAVE_STORE_PATH, "w", encoding="utf-8") as store_file:
            json.dump(store, store_file, indent=2, default=str)
        return True
    except Exception as error:
        print(f"[LEAVE] Couldn't write store at {LEAVE_STORE_PATH}: {error}", flush=True)
        return False


def _leave_user_key(user):
    """
    Normalizes a user name/email to the key used in the leave store.
    Reuses resolve_calendar_user() so "Priya" and "priya sharma"
    (configured as a calendar user) land on the same key; falls back
    to the lowercased literal name for users with no calendar
    configured at all.
    """

    return (resolve_calendar_user(user) or (user or "")).strip().lower()


def get_leave_eligible_users():
    """
    Set of lowercased user names allowed to take leave, or None if
    NOVA_LEAVE_ELIGIBLE_USERS isn't configured (meaning: no
    restriction, everyone is eligible).
    """

    raw = os.environ.get("NOVA_LEAVE_ELIGIBLE_USERS", "")

    if not raw.strip():
        return None

    return {name.strip().lower() for name in raw.split(",") if name.strip()}


def is_leave_eligible(user):
    """True if `user` is allowed to apply for leave."""

    eligible = get_leave_eligible_users()

    if eligible is None:
        return True

    return _leave_user_key(user) in eligible


def get_leave_balances(user):
    """
    Full {leave_type: days_remaining} dict for `user`, merged with
    DEFAULT_LEAVE_BALANCES for any leave type never explicitly set.
    """

    store = _load_leave_store()
    user_key = _leave_user_key(user)
    saved = store.get("balances", {}).get(user_key, {})

    merged = dict(DEFAULT_LEAVE_BALANCES)
    merged.update(saved)
    return merged


def get_leave_balance(user, leave_type):
    """Days remaining of `leave_type` for `user`."""

    return get_leave_balances(user).get(
        leave_type, DEFAULT_LEAVE_BALANCES.get(leave_type, 0)
    )


def get_leave_history(user, include_cancelled=False):
    """All recorded leave requests for `user`, most recent first."""

    store = _load_leave_store()
    user_key = _leave_user_key(user)

    history = [
        request
        for request in store.get("requests", [])
        if request.get("user") == user_key
        and (include_cancelled or request.get("status") != "cancelled")
    ]

    history.sort(key=lambda r: r.get("start", ""), reverse=True)
    return history


def check_duplicate_leave(user, start_date, end_date):
    """
    Finds any existing (non-cancelled) leave request for `user` that
    overlaps [start_date, end_date] (inclusive date objects).

    Returns a list of the overlapping request records.
    """

    overlaps = []

    for request in get_leave_history(user):
        try:
            request_start = datetime.strptime(request["start"], "%Y-%m-%d").date()
            request_end = datetime.strptime(request["end"], "%Y-%m-%d").date()
        except Exception:
            continue

        if request_start <= end_date and request_end >= start_date:
            overlaps.append(request)

    return overlaps


def validate_leave_request(user, leave_type, start_date, end_date, reason=""):
    """
    Runs every pre-submission validation for a leave request without
    writing anything - safe to call just to preview/confirm.

    Args:
        user: name/email of the person requesting leave.
        leave_type: e.g. "annual", "sick", "casual".
        start_date, end_date: date or datetime, inclusive range.
        reason: optional free-text reason (not validated, just
            carried through to `info` for the confirmation card).

    Returns:
        ok: True only if there are no blocking errors.
        errors: list of blocking problems - submission should be
            refused (or require an explicit override) while any of
            these are present.
        warnings: list of non-blocking notes worth surfacing to the
            user before they confirm (e.g. a calendar clash).
        info: dict of computed details (working_days, balance_before,
            duplicates, calendar_conflicts, ...) useful for building
            a confirmation card or receipt.
    """

    errors = []
    warnings = []
    info = {"leave_type": leave_type, "reason": reason}

    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    if start_date is None or end_date is None:
        errors.append("A start and end date are both required.")
        return False, errors, warnings, info

    # ---- date range validity ----
    if end_date < start_date:
        errors.append(
            f"The end date ({end_date}) is before the start date "
            f"({start_date})."
        )
        return False, errors, warnings, info

    span_days = (end_date - start_date).days + 1

    if span_days > LEAVE_MAX_SPAN_DAYS:
        errors.append(
            f"A single leave request can't span more than "
            f"{LEAVE_MAX_SPAN_DAYS} days (this one is {span_days})."
        )

    # ---- past dates ----
    today = datetime.now().date()

    if start_date < today:
        errors.append(
            f"The start date ({start_date}) is in the past - leave "
            "can't be requested for dates that have already gone by."
        )

    # ---- eligibility ----
    if not is_leave_eligible(user):
        errors.append(
            f"{user} isn't on the configured list of leave-eligible "
            "users."
        )

    # ---- weekends / holidays ----
    all_days = list(_daterange(start_date, end_date))
    working_days = [
        day for day in all_days if not is_weekend(day) and not is_holiday(day)
    ]
    non_working_days = len(all_days) - len(working_days)

    info["working_days"] = len(working_days)
    info["non_working_days"] = non_working_days

    if not working_days:
        errors.append(
            "The requested range falls entirely on weekends/holidays "
            "- there are no working days to apply leave for."
        )
    elif non_working_days:
        warnings.append(
            f"{non_working_days} day(s) in this range are weekends/"
            "holidays and won't be deducted from your balance."
        )

    # ---- leave balance ----
    balance = get_leave_balance(user, leave_type)
    info["balance_before"] = balance
    info["days_requested"] = len(working_days)

    if working_days and len(working_days) > balance:
        errors.append(
            f"Insufficient {leave_type} leave balance: {balance} "
            f"day(s) available, {len(working_days)} requested."
        )

    # ---- duplicate / overlapping requests ----
    duplicates = check_duplicate_leave(user, start_date, end_date)
    info["duplicates"] = duplicates

    if duplicates:
        dup_text = "; ".join(
            f"{d.get('leave_type', 'leave')} {d['start']} to {d['end']}"
            for d in duplicates
        )
        errors.append(f"Overlapping leave request already exists: {dup_text}.")

    # ---- existing calendar events during the leave window ----
    # Non-blocking - a meeting during requested leave is worth
    # flagging (so the user can reschedule or note a handover) but
    # shouldn't by itself stop the leave from being applied for.
    info["calendar_conflicts"] = []

    if Calendar is not None:
        try:
            window_start = datetime.combine(start_date, datetime.min.time())
            window_end = datetime.combine(end_date, datetime.max.time())

            conflicts, _unchecked = check_group_availability(
                user, window_start, window_end
            )

            matched_user = resolve_calendar_user(user)

            if matched_user and conflicts.get(matched_user):
                info["calendar_conflicts"] = conflicts[matched_user]
                warnings.append(
                    "You have existing calendar event(s) during this "
                    f"period: {'; '.join(conflicts[matched_user])}."
                )
        except Exception as error:
            print(f"[LEAVE] calendar-conflict check failed: {error}", flush=True)

    ok = not errors
    return ok, errors, warnings, info


def _format_leave_request_email(user_key, request_record, warnings):
    """
    Builds the subject/body for the leave-request email sent to the
    approver - the human-readable request they'll actually approve
    or reject (by whatever process the approval flow ends up being;
    not built yet - see the module note above apply_leave()).
    """

    # Never show the literal internal "me" key to the approver -
    # fall back to a neutral placeholder if no real name is
    # configured (see NOVA_LEAVE_USER_NAME above).
    if user_key == "me":
        display_name = LEAVE_USER_DISPLAY_NAME or "Team Member"
    else:
        display_name = LEAVE_USER_DISPLAY_NAME or user_key.title()

    leave_type_label = f"{request_record['leave_type'].title()} leave"

    subject = (
        f"Leave request - {display_name} - {request_record['leave_type'].title()} "
        f"({request_record['start']} to {request_record['end']})"
    )

    lines = [
        f"{display_name} has requested leave:",
        "",
        f"Type: {request_record['leave_type'].title()}",
        f"Dates: {request_record['start']} to {request_record['end']}",
        f"Working days: {request_record['days']}",
        # The reason line is intentionally a fixed "<Type> leave"
        # label (e.g. "Sick leave") rather than whatever free text
        # got captured during extraction - that free text is often
        # just the original request restated ("apply a sick leave on
        # monday"), not an actual reason, so a clean type-based line
        # reads better to an approver. The real free-text reason (if
        # any) is still kept on the stored record/history for the
        # requester's own reference - only the email display is
        # normalized here.
        f"Reason: {leave_type_label}",
    ]

    if warnings:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"- {warning}" for warning in warnings)

    lines.append("")
    lines.append(
        f"Request ID: {request_record['id']} (status: {request_record['status']})"
    )

    return subject, "\n".join(lines)


def apply_leave(user, leave_type, start_date, end_date, reason="", force=False):
    """
    Validates a leave request (via validate_leave_request()) and, if
    it passes, records it as PENDING and emails the configured leave
    approver (NOVA_LEAVE_APPROVER_EMAIL) to request their sign-off.

    This is deliberately just the REQUEST half of the flow - the
    balance is NOT deducted here, and the record stays "pending"
    indefinitely. Approving/rejecting a pending request (which is
    what should actually move it to "approved" and deduct the
    balance, or to "rejected" and release it) is a separate piece of
    work, not built yet.

    Args:
        force: if True, skips the "ok" check and records/sends the
            request anyway (used when a user has already seen and
            explicitly confirmed past a validation warning-only
            flow; blocking errors should still generally be fixed
            rather than forced through).

    Returns:
        (success: bool, message: str, details: dict) - `message` is
        short and user-facing; `details` carries errors/warnings/the
        stored record, plus whether the approval email actually
        went out.
    """

    ok, errors, warnings, info = validate_leave_request(
        user, leave_type, start_date, end_date, reason
    )

    if not ok and not force:
        return (
            False,
            "Leave request couldn't be submitted: " + " ".join(errors),
            {"errors": errors, "warnings": warnings, "info": info},
        )

    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    store = _load_leave_store()
    user_key = _leave_user_key(user)

    # Balance is intentionally NOT touched here - it's only ever
    # read (see get_leave_balance()) until an approval step exists
    # to actually deduct it. Requesting leave should hold a place in
    # the request log, not silently spend days that might still be
    # rejected.
    store["balances"].setdefault(user_key, dict(DEFAULT_LEAVE_BALANCES))

    days_requested = info.get("days_requested", 0)

    request_record = {
        "id": str(uuid.uuid4()),
        "user": user_key,
        "leave_type": leave_type,
        "start": str(start_date),
        "end": str(end_date),
        "days": days_requested,
        "reason": reason,
        "status": "pending",
        "requested_at": datetime.now().isoformat(timespec="seconds"),
    }
    store["requests"].append(request_record)

    if not _save_leave_store(store):
        return (
            False,
            "Couldn't save the leave request - please try again.",
            {"errors": errors, "warnings": warnings, "info": info},
        )

    print(
        f"[LEAVE] {user_key}: {leave_type} {request_record['start']} to "
        f"{request_record['end']} ({days_requested} day(s)) - pending approval",
        flush=True,
    )

    # ---- email the approver ----
    mail_sent = False
    mail_message = ""

    if not LEAVE_APPROVER_EMAIL:
        mail_message = (
            "No leave approver is configured "
            "(set NOVA_LEAVE_APPROVER_EMAIL) - the request was saved "
            "but nobody was emailed."
        )
    else:
        subject, body = _format_leave_request_email(user_key, request_record, warnings)
        mail_sent, mail_message = send_mail(to=LEAVE_APPROVER_EMAIL, subject=subject, body=body)

        if not mail_sent:
            mail_message = f"The request was saved, but the approval email failed to send: {mail_message}"

    base = (
        f"{leave_type.title()} leave request from {request_record['start']} "
        f"to {request_record['end']} ({days_requested} working day(s)) "
        f"for {user_key}"
    )

    if mail_sent:
        message = (
            f"{base} was sent to your leave approver for sign-off - "
            "it's pending until they approve it."
        )
    else:
        message = f"{base} was saved as pending. {mail_message}".strip()

    if warnings:
        message += " Note: " + " ".join(warnings)

    return (
        True,
        message,
        {
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "record": request_record,
            "approver_notified": mail_sent,
        },
    )


def get_pending_leave_requests():
    """
    All leave requests currently awaiting approval, across every
    user, oldest first - what an approver needs to work through.
    """

    store = _load_leave_store()

    pending = [
        request
        for request in store.get("requests", [])
        if request.get("status") == "pending"
    ]

    pending.sort(key=lambda r: r.get("requested_at", ""))
    return pending


def get_all_leave_requests():
    """
    Every leave request ever submitted, across every user and every
    status (pending/approved/rejected), most recently requested
    first.

    Used by the sidebar's read-only leave status list: unlike
    get_pending_leave_requests(), requests don't disappear from this
    list once they're approved/rejected - they just show a different
    status. Approval itself isn't done from here; that happens
    elsewhere (see approve_leave_request()/reject_leave_request()).
    """

    store = _load_leave_store()

    requests = list(store.get("requests", []))
    requests.sort(key=lambda r: r.get("requested_at", ""), reverse=True)
    return requests


def _find_leave_request(store, request_id):
    """Returns the request dict with this id, or None."""

    for request in store.get("requests", []):
        if request.get("id") == request_id:
            return request
    return None


def approve_leave_request(request_id, approver_note=""):
    """
    Approves a pending leave request: marks it "approved", deducts
    its working days from the requester's balance for that leave
    type (held until now - see apply_leave()), and emails the
    requester's own mailbox (NOVA_SMTP_USER - the account NOVA sends
    as, which is also whoever submitted the request) to confirm.

    Returns:
        (success: bool, message: str) - short, user-facing (shown to
        whoever clicked Approve, in the admin sidebar).
    """

    store = _load_leave_store()
    request = _find_leave_request(store, request_id)

    if request is None:
        return False, "That leave request no longer exists."

    if request.get("status") != "pending":
        return False, f"That request is already {request.get('status')}."

    user_key = request["user"]
    leave_type = request["leave_type"]
    days = request.get("days", 0)

    user_balances = store["balances"].setdefault(user_key, dict(DEFAULT_LEAVE_BALANCES))
    for lt, default_amount in DEFAULT_LEAVE_BALANCES.items():
        user_balances.setdefault(lt, default_amount)

    current_balance = user_balances.get(
        leave_type, DEFAULT_LEAVE_BALANCES.get(leave_type, 0)
    )

    if days > current_balance:
        # Balance may have shifted since the request was submitted
        # (e.g. another request for the same user/type got approved
        # first) - refuse rather than letting the balance go
        # negative silently.
        return False, (
            f"Can't approve - {user_key} only has {current_balance} "
            f"{leave_type} day(s) left, but this request needs {days}."
        )

    user_balances[leave_type] = current_balance - days
    request["status"] = "approved"
    request["approved_at"] = datetime.now().isoformat(timespec="seconds")
    if approver_note:
        request["approver_note"] = approver_note

    if not _save_leave_store(store):
        return False, "Couldn't save the approval - please try again."

    print(
        f"[LEAVE] {user_key}: {leave_type} {request['start']} to "
        f"{request['end']} - APPROVED ({days} day(s) deducted)",
        flush=True,
    )

    if SMTP_USER:
        subject = f"Leave approved - {request['start']} to {request['end']}"
        body = (
            f"Your {leave_type} leave request for {request['start']} to "
            f"{request['end']} ({days} working day(s)) has been approved.\n\n"
            f"Remaining {leave_type} balance: {user_balances[leave_type]}."
        )
        if approver_note:
            body += f"\n\nNote from approver: {approver_note}"
        send_mail(to=SMTP_USER, subject=subject, body=body)

    return True, (
        f"Approved {leave_type} leave for {user_key} "
        f"({request['start']} to {request['end']}, {days} day(s))."
    )


def reject_leave_request(request_id, approver_note=""):
    """
    Rejects a pending leave request: marks it "rejected" (balance
    was never deducted for a pending request - see apply_leave() -
    so there's nothing to release) and emails the requester's own
    mailbox to let them know.

    Returns:
        (success: bool, message: str)
    """

    store = _load_leave_store()
    request = _find_leave_request(store, request_id)

    if request is None:
        return False, "That leave request no longer exists."

    if request.get("status") != "pending":
        return False, f"That request is already {request.get('status')}."

    request["status"] = "rejected"
    request["rejected_at"] = datetime.now().isoformat(timespec="seconds")
    if approver_note:
        request["approver_note"] = approver_note

    if not _save_leave_store(store):
        return False, "Couldn't save the rejection - please try again."

    print(
        f"[LEAVE] {request['user']}: {request['leave_type']} "
        f"{request['start']} to {request['end']} - REJECTED",
        flush=True,
    )

    if SMTP_USER:
        subject = f"Leave rejected - {request['start']} to {request['end']}"
        body = (
            f"Your {request['leave_type']} leave request for "
            f"{request['start']} to {request['end']} was not approved."
        )
        if approver_note:
            body += f"\n\nNote from approver: {approver_note}"
        send_mail(to=SMTP_USER, subject=subject, body=body)

    return True, (
        f"Rejected {request['leave_type']} leave for {request['user']} "
        f"({request['start']} to {request['end']})."
    )