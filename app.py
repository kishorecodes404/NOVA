"""
Streamlit AI Chatbot using Google Gemini.

Install:
    pip install streamlit google-genai

Set API key in PowerShell:
    $env:GEMINI_API_KEY="your_gemini_api_key"

Or add it to a .env file next to this script:
    GEMINI_API_KEY=your_gemini_api_key

Mail agent (read: NOVA_IMAP_*, send: NOVA_SMTP_*) and meetings
agent (NOVA_MEETINGS_ICS_PATH) are configured via env vars too -
see the docstrings in rag.py above search_mail(), send_mail(),
search_meetings() and schedule_meeting() for details.

Run:
    streamlit run app.py
"""

import json
import os
import random
import re
import requests
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

from rag import (
    retrieve_context,
    clear_documents,
    index_admin_document,
    web_search,
    search_mail,
    search_meetings,
    search_meetings_multi,
    send_mail,
    schedule_meeting,
    check_group_availability,
    resolve_calendar_user,
    get_configured_calendar_users,
    validate_leave_request,
    apply_leave,
    get_leave_balance,
    get_leave_balances,
    get_leave_history,
    get_all_leave_requests,
    clear_leave_requests,
    validate_po_request,
    apply_po,
    get_po_history,
    get_all_po_requests,
    get_sent_po_requests,
    approve_po_request,
    reject_po_request,
    clear_po_requests,
    EMBEDDING_MODEL,
    list_indexed_documents,
)

# ---------------------------------------------------
# Timing log
#
# Writes straight to a file next to app.py, so it's findable
# no matter how Streamlit is being run (terminal, Docker,
# background process, etc). Each line is also appended with
# a flush, so nothing sits in a buffer waiting to be written.
# ---------------------------------------------------

TIMING_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "nova_timing.log",
)


def log_timing(message):
    line = f"{time.strftime('%H:%M:%S')} | {message}"
    print(line, flush=True)
    try:
        with open(TIMING_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")
            log_file.flush()
    except Exception:
        pass

# Configuration
# ---------------------------------------------------
APP_NAME = "NOVA"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_SESSION = requests.Session()

# Groq - OpenAI-compatible endpoint, hosted inference on open-weight
# models. Used as a fast/accurate alternative when local Ollama
# answers aren't accurate enough and Gemini's free-tier quota is
# unreliable. openai/gpt-oss-120b is Groq's current flagship
# general-purpose model (llama-3.3-70b-versatile was deprecated by
# Groq in June 2026, migrated per Groq's own guidance).
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_SESSION = requests.Session()
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

# Two different local models, doing two different jobs:
#
# ROUTER_MODEL  - classifies DOCUMENT/WEB/CHAT and writes the 3
#                 follow-up questions. Both are small, cheap,
#                 latency-sensitive tasks - a tiny model is not a
#                 quality bottleneck here, so keep it fast.
#
# ANSWER_MODEL  - actually writes the response the user reads.
#                 This is where model quality matters. Bump this
#                 up to whatever your GPU comfortably holds:
#                   ~6-8GB VRAM  -> "qwen2.5:3b-instruct"
#                   ~8-12GB VRAM -> "qwen2.5:7b-instruct"   (default)
#                   16GB+ VRAM   -> "qwen2.5:14b-instruct"
#                 Pull it first: `ollama pull qwen2.5:7b-instruct`
#
# Both models stay resident (keep_alive: 24h) so there's no
# per-turn reload cost - but Ollama needs to be allowed to hold
# more than one model in memory at once, otherwise it'll swap
# ROUTER_MODEL and ANSWER_MODEL in and out on every single turn
# (the ~7-10s reload penalty). Before starting the Ollama server:
#
#     OLLAMA_MAX_LOADED_MODELS=3   (router + answer + embedding)
#
# and make sure combined VRAM for all three models fits on the GPU.
ROUTER_MODEL = "qwen2.5:1.5b"
ANSWER_MODEL = "qwen2.5:3b"

# Kept for backwards compatibility with code/UI that just needs to
# know "is this local mode or Gemini mode" - not tied to a specific
# model string anymore.
LOCAL_MODEL_NAME = "qwen2.5:1.5b"

# Every call to the SAME model below uses this SAME num_ctx. Ollama
# can treat a different num_ctx as a different context and reload
# the model to switch - mixing context sizes across calls to the
# same model was causing reloads on nearly every request (~7-10s
# each time). ROUTER_MODEL and ANSWER_MODEL are different models
# regardless, so they're already cached separately by Ollama; this
# constant just needs to stay consistent within each one.
#
# 768 was too small: a WEB-routed prompt (rules + 4 search
# results) already runs ~500-600 tokens on its own, leaving next
# to no room for the answer. When that budget overflows, Ollama
# drops tokens from the FRONT of the prompt - which is exactly
# where the "don't invent information" rules live - so the model
# was answering ungrounded. 2048 gives real headroom.
LOCAL_MODEL_NUM_CTX = 2048

@st.cache_resource
def warm_up_ollama():
    for model in (ROUTER_MODEL, ANSWER_MODEL):
        try:
            OLLAMA_SESSION.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": "Hi",
                    "stream": False,
                    "keep_alive": "24h",
                    "options": {
                        "num_predict": 1,
                        "num_ctx": LOCAL_MODEL_NUM_CTX,
                        "num_gpu": 99,
                    },
                },
                timeout=60,
            )
        except Exception:
            pass


@st.cache_resource
def warm_up_embedding_model():
    """
    Load the embedding model at startup too, so the first
    document-routed question doesn't pay a cold-load cost on
    top of the chat model's.

    Note: if the GPU can't hold both qwen2.5:1.5b and
    qwen3-embedding:0.6b resident at the same time, Ollama will
    still evict one to load the other on every DOCUMENT-routed
    turn - this warm-up only helps the very first call, not the
    ongoing swap. That needs a hardware/Ollama-config fix (see
    OLLAMA_MAX_LOADED_MODELS / available VRAM), not app code.
    """
    try:
        from rag import create_embedding
        create_embedding("warm up")
    except Exception:
        pass

SYSTEM_INSTRUCTION = """
You are NOVA, a helpful and thoughtful AI assistant.
Give clear, accurate, and concise answers.
Use Markdown when it improves readability.
"""


st.set_page_config(
    page_title=f"{APP_NAME} · AI Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# EMAIL PO APPROVAL HANDLER
#
# Gmail/Outlook approval buttons open NOVA with query parameters.
# No public API server is required: Streamlit handles the request
# when the page loads. The signed token prevents someone from
# approving/rejecting a different PO by changing the URL.
# ============================================================

def _handle_po_email_action():
    try:
        params = st.query_params
        action = params.get("po_action", "")
        request_id = params.get("po_id", "")
        token = params.get("po_token", "")

        if action not in {"approve", "reject"} or not request_id or not token:
            return False

        from rag import verify_po_approval_token

        if not verify_po_approval_token(request_id, action, token):
            st.error("This PO approval link is invalid or has been tampered with.")
            return True

        st.title("NOVA · Purchase Order Approval")

        if action == "approve":
            ok, message = approve_po_request(request_id, "Approved from email.")
            if ok:
                st.success(message)
                st.info("The PO has been sent to the vendor.")
            else:
                st.error(message)
        else:
            ok, message = reject_po_request(request_id, "Rejected from email.")
            if ok:
                st.warning(message)
                st.info("The vendor was not emailed.")
            else:
                st.error(message)

        st.caption("You can close this tab. The approval action has been recorded in NOVA.")
        return True
    except Exception as error:
        st.error(f"Could not process the PO approval action: {error}")
        return True


# Handle one-click PO approval/rejection links from email only after
# Streamlit page configuration has been initialized (and after the
# handler function above has actually been defined).
if _handle_po_email_action():
    st.stop()


# ---------------------------------------------------
# Styling
# ---------------------------------------------------
def apply_custom_styles():
    st.markdown(
        """
        <style>

        /* ================================
           NOVA — GLOBAL
        ================================= */

        :root {
            --bg: #f5f5f7;
            --sidebar: #18181c;
            --sidebar-card: #222228;
            --text: #1d1d1f;
            --muted: #73737c;
            --white: #ffffff;
            --purple: #6d45e8;
            --purple-light: #8064f2;
            --border: rgba(0,0,0,0.08);
        }

        .stApp {
            background: var(--bg) !important;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--bg) !important;
        }

        [data-testid="stMain"] {
            background: var(--bg) !important;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        .block-container {
            max-width: 1100px !important;
            padding-top: 2rem !important;
            padding-bottom: 8rem !important;
        }


        /* ================================
           SIDEBAR
        ================================= */

        [data-testid="stSidebar"] {
            background: #202025 !important;
            border-right: 1px solid rgba(0,0,0,0.08);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 1.25rem 1rem !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 1.5rem 1rem !important;
        }

       
        .nova-sidebar-logo {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #f2f2f4 !important;
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            margin-bottom: 2px;
        }

        .nova-sidebar-star {
            color: var(--purple-light) !important;
            font-size: 2rem;
        }

        .nova-sidebar-subtitle {
            color: #85858e !important;
            font-size: 0.82rem;
            margin-left: 2px;
            margin-bottom: 1.3rem;
        }

        .sidebar-section-title {
            color: #85858e !important;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin: 1.25rem 0 0.55rem;
        }


        /* =========================================
           NOVA SIDEBAR — CONTROL OVERRIDE
        ========================================= */

        /* ---------- AI MODE SELECTBOX ---------- */

        [data-testid="stSidebar"] div[data-baseweb="select"] {
            width: 100% !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            width: 100% !important;
            min-height: 42px !important;

            background: #29292f !important;
            background-color: #29292f !important;

            border: 1px solid rgba(255,255,255,0.09) !important;
            border-radius: 11px !important;

            box-shadow: none !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] * {
            color: #eeeeef !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] svg {
            fill: #85858e !important;
            color: #85858e !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] > div > div {
            background: transparent !important;
        }

        /* ---------- DROPDOWN MENU ---------- */

        div[data-baseweb="popover"] {
            background: #1b1b20 !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
        }

        div[data-baseweb="popover"] [role="option"] {
            background: #1b1b20 !important;
            color: #eeeeef !important;
        }

        div[data-baseweb="popover"] [role="option"]:hover {
            background: #292931 !important;
        }



        /* ---------- SIDEBAR BUTTONS ---------- */

        [data-testid="stSidebar"] .stButton {
            width: 100% !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100% !important;
            min-height: 42px !important;

            background: #29292f !important;
            background-color: #29292f !important;
            color: #eeeeef !important;

            border: 1px solid rgba(255,255,255,0.09) !important;
            border-radius: 11px !important;

            box-shadow: none !important;

            font-size: 0.88rem !important;
            font-weight: 500 !important;

            transition:
                background-color 0.18s ease,
                border-color 0.18s ease,
                transform 0.15s ease;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: #323239 !important;
            background-color: #323239 !important;
            color: #ffffff !important;
            border-color: rgba(128,100,242,0.35) !important;
        }

        [data-testid="stSidebar"] .stButton > button:active {
            transform: scale(0.985);
        }

        /* ---------- FOCUS ---------- */

        [data-testid="stSidebar"] input:focus,
        [data-testid="stSidebar"] [data-baseweb="select"]:focus-within {
            outline: none !important;
            box-shadow: 0 0 0 1px rgba(128,100,242,0.5) !important;
        }

        /* ---------- SIDEBAR EXPANDER (e.g. "Indexed Documents") ---------- */

        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary span {
            color: #eeeeef !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
            fill: #eeeeef !important;
        }

        /* ================================
           ADMIN UPLOADER
        ================================= */

        [data-testid="stSidebar"] [data-testid="stFileUploader"] {
            width: 100% !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
            background: #29292f !important;
            border: 1px dashed rgba(255,255,255,0.16) !important;
            border-radius: 11px !important;
            padding: 0.8rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] section > div {
            background: transparent !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] small {
            color: #85858e !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
            background: #323239 !important;
            color: #eeeeef !important;
            border: 1px solid rgba(255,255,255,0.09) !important;
            border-radius: 8px !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
            background: #3a3a42 !important;
            border-color: rgba(128,100,242,0.35) !important;
        }



        /* ================================
           MAIN HEADER
        ================================= */

        .nova-topbar {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 3rem;
        }

        .nova-top-logo {
            color: #5b38c9;
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.04em;
        }

        .nova-top-logo span {
            color: var(--purple);
            margin-right: 8px;
            font-size: 2.5rem;
        }

        /* ================================
           EMPTY STATE
        ================================= */

        .nova-empty-state {
            min-height: 58vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding-bottom: 2rem;
        }

        .nova-empty-star {
            color: var(--purple);
            font-size: 1.7rem;
            margin-bottom: 0.8rem;
        }

        .nova-empty-state h1 {
            color: var(--text);
            font-size: clamp(2.8rem, 5vw, 4rem);
            font-weight: 750;
            letter-spacing: -0.06em;
            line-height: 1;
            margin: 0;
        }

        .nova-empty-state p {
            color: var(--muted);
            font-size: 1.05rem;
            margin-top: 1rem;
        }       

        /* ================================
           CHAT
        ================================= */

        [data-testid="stChatMessage"] {
            border-radius: 18px !important;
            border: 1px solid var(--border) !important;
            background: white !important;
            box-shadow: 0 5px 20px rgba(0,0,0,0.04) !important;
            padding: 0.8rem 1rem !important;
            margin: 1rem 0 !important;
        }

        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li {
            color: var(--text) !important;
        }


        /* ================================
           CHAT INPUT
        ================================= */

        [data-testid="stChatInput"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }

        [data-testid="stChatInput"] > div {
            background: white !important;
            border: 1px solid rgba(0,0,0,0.1) !important;
            border-radius: 22px !important;
            box-shadow: 0 8px 30px rgba(0,0,0,0.08) !important;
            padding: 0.4rem 0.55rem !important;
        }

        [data-testid="stChatInput"] textarea {
            color: var(--text) !important;
            background: transparent !important;
            font-size: 1rem !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #888894 !important;
        }

        [data-testid="stChatInput"] button {
            background: var(--purple) !important;
            color: white !important;
            border-radius: 50% !important;
        }

        [data-testid="stChatInput"] button:hover {
            background: var(--purple-light) !important;
        }


        /* ================================
           FOLLOW-UP BUTTONS
        ================================= */

        .followup-title {
            color: #24242a;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }

        .st-key-followup_row .stButton > button {
            background: white !important;
            border: 1px solid rgba(109,69,232,0.18) !important;
            color: var(--purple) !important;
            border-radius: 12px !important;
        }


        /* ================================
           LOADER
        ================================= */

        .nova-loader {
            width: 20px;
            height: 20px;
            border: 3px solid rgba(109,69,232,0.18);
            border-top-color: var(--purple);
            border-radius: 50%;
            animation: nova-spin 0.8s linear infinite;
            display: inline-block;
            margin-right: 10px;
            vertical-align: middle;
        }

        @keyframes nova-spin {
            to {
                transform: rotate(360deg);
            }
        }


        /* ================================
           FOOTER
        ================================= */

        .nova-footer {
            text-align: center;
            color: #8a8a93;
            font-size: 0.72rem;
            margin-top: 0.7rem;
        }
        /* Hide Streamlit text-input instruction */
        [data-testid="InputInstructions"] {
            display: none !important;
        }

        /* ================================
           ADMIN KNOWLEDGE BASE
        ================================= */

        [data-testid="stSidebar"] [data-testid="stFileUploader"] label {
            color: #eeeeef !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] small {
            color: #b8b8c2 !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] span {
            color: #eeeeef !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] p {
            color: #b8b8c2 !important;
        }

        /* ================================
           SIDEBAR COLLAPSE BUTTON
        ================================= */

        [data-testid="stSidebar"] button[kind="headerNoPadding"] {
            color: #ffffff !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] button[kind="headerNoPadding"] *,
        [data-testid="stSidebar"] button[kind="headerNoPadding"] svg,
        [data-testid="stSidebar"] button[kind="headerNoPadding"] svg path {
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] button[kind="headerNoPadding"]:hover {
            background: rgba(255,255,255,0.08) !important;
        }

/* =========================================
   CHATGPT-STYLE CONVERSATION LIST
========================================= */

/* Conversation list container */
[data-testid="stSidebar"] .st-key-conversation_list {
    margin: 0 !important;
    padding: 0 !important;
}

/* Remove spacing around conversation buttons */
[data-testid="stSidebar"] .st-key-conversation_list .stButton {
    margin: 0 !important;
    padding: 0 !important;
}

/* Conversation button */
[data-testid="stSidebar"] .st-key-conversation_list .stButton > button {
    width: 100% !important;

    min-height: 36px !important;
    height: 36px !important;

    margin: 1px 0 !important;
    padding: 0 10px !important;

    background: transparent !important;
    background-color: transparent !important;

    border: 1px solid transparent !important;
    border-radius: 8px !important;

    color: #eeeeef !important;

    font-size: 0.86rem !important;
    font-weight: 400 !important;

    text-align: left !important;
    justify-content: flex-start !important;

    box-shadow: none !important;

    overflow: hidden !important;

    transition:
        background-color 0.15s ease,
        color 0.15s ease !important;
}

/* Remove Streamlit's internal button backgrounds */
[data-testid="stSidebar"] .st-key-conversation_list .stButton > button > div {
    background: transparent !important;
    background-color: transparent !important;

    border: none !important;
    box-shadow: none !important;

    width: 100% !important;

    margin: 0 !important;
    padding: 0 !important;
}

/* Conversation text */
[data-testid="stSidebar"] .st-key-conversation_list .stButton > button p {
    color: #eeeeef !important;

    background: transparent !important;
    background-color: transparent !important;

    width: 100% !important;

    margin: 0 !important;
    padding: 0 !important;

    font-size: 0.86rem !important;
    font-weight: 400 !important;

    text-align: left !important;

    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Hover */
[data-testid="stSidebar"] .st-key-conversation_list .stButton > button:hover {
    background: #29292f !important;
    background-color: #29292f !important;

    border: 1px solid transparent !important;

    color: #ffffff !important;
}

/* Keep contents transparent on hover */
[data-testid="stSidebar"] .st-key-conversation_list .stButton > button:hover > div,
[data-testid="stSidebar"] .st-key-conversation_list .stButton > button:hover p {
    background: transparent !important;
    background-color: transparent !important;
}

/* =========================================
   ACTIVE CONVERSATION
========================================= */

[data-testid="stSidebar"] .st-key-conversation_list .stButton > button.active-chat {
    background: #29292f !important;
    background-color: #29292f !important;

    border-color: transparent !important;
}

/* Active text */
[data-testid="stSidebar"] .st-key-conversation_list .stButton > button.active-chat p {
    color: #ffffff !important;
    font-weight: 500 !important;
}

/* =========================================
   CONVERSATION LIST — LEFT ALIGNMENT
========================================= */

[data-testid="stSidebar"] .st-key-conversation_list .stButton > button {
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: center !important;

    padding-left: 10px !important;
}

[data-testid="stSidebar"] .st-key-conversation_list .stButton > button > div {
    width: 100% !important;

    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;

    text-align: left !important;

    margin: 0 !important;
    padding: 0 !important;

    background: transparent !important;
}

[data-testid="stSidebar"] .st-key-conversation_list .stButton > button p {
    width: 100% !important;

    margin: 0 !important;
    padding: 0 !important;

    text-align: left !important;

    color: #eeeeef !important;

    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------
# Gemini and chat memory functions
# ---------------------------------------------------
def get_api_key():
    """Get API key from .env, Streamlit secrets, or environment variables."""
    try:
        return st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    except FileNotFoundError:
        return os.getenv("GEMINI_API_KEY")


def get_groq_api_key():
    """
    Get the Groq API key from .env, secrets, or env vars.

    Checks GROQ_API_KEY first (the name anyone would naturally use
    for a Groq key), then falls back to GROK_API_KEY (no Q) for
    backwards compatibility with however this was originally set up.
    A 403 from Groq with a key that "should" work is almost always
    this: the key got set under a name this function wasn't reading.
    """
    try:
        return (
            st.secrets.get("GROQ_API_KEY")
            or st.secrets.get("GROK_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GROK_API_KEY")
        )
    except FileNotFoundError:
        return os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")


CHAT_DB = "chat_history.db"


def init_chat_database():
    """Create and migrate the chat history database."""

    connection = sqlite3.connect(CHAT_DB)
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        "PRAGMA table_info(messages)"
    )

    columns = [
        column[1]
        for column in cursor.fetchall()
    ]

    if "chat_id" not in columns:

        cursor.execute(
            """
            ALTER TABLE messages
            ADD COLUMN chat_id INTEGER
            """
        )

    # Persist the routed agent (MAIL/MEETINGS/etc) and its evidence
    # sources alongside each assistant message. Previously these
    # only lived in transient session_state (last_route/
    # last_sources), which the history-redraw loop never read - so
    # the agent badge and Sources expander disappeared for every
    # message except the one just generated, on the very next
    # rerun/reload.
    if "route" not in columns:

        cursor.execute(
            """
            ALTER TABLE messages
            ADD COLUMN route TEXT
            """
        )

    if "sources" not in columns:

        cursor.execute(
            """
            ALTER TABLE messages
            ADD COLUMN sources TEXT
            """
        )

    connection.commit()
    connection.close()

def migrate_old_messages():
    """Attach old messages to a conversation."""

    connection = sqlite3.connect(CHAT_DB)
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM messages
        WHERE chat_id IS NULL
        ORDER BY id
        """
    )

    old_messages = cursor.fetchall()

    if not old_messages:
        connection.close()
        return

    cursor.execute(
        """
        SELECT id
        FROM chats
        ORDER BY id
        LIMIT 1
        """
    )

    existing_chat = cursor.fetchone()

    if existing_chat:
        chat_id = existing_chat[0]

    else:
        cursor.execute(
            """
            INSERT INTO chats (title)
            VALUES (?)
            """,
            ("Previous Conversation",),
        )

        chat_id = cursor.lastrowid

    cursor.execute(
        """
        UPDATE messages
        SET chat_id = ?
        WHERE chat_id IS NULL
        """,
        (chat_id,),
    )

    connection.commit()
    connection.close()

def create_chat(title="Untitled Chat"):
    """Create a new conversation and return its ID."""

    connection = sqlite3.connect(CHAT_DB)
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO chats (title)
        VALUES (?)
        """,
        (title,),
    )

    chat_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return chat_id

def get_chats():
    """Load all saved conversations."""
    connection = sqlite3.connect(CHAT_DB)
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, title, created_at
        FROM chats
        ORDER BY created_at DESC
        """
    )

    chats = cursor.fetchall()
    connection.close()

    return chats


def load_chat_messages(chat_id):
    """Load all messages belonging to one conversation."""
    connection = sqlite3.connect(CHAT_DB)
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT role, content, route, sources
        FROM messages
        WHERE chat_id = ?
        ORDER BY id
        """,
        (chat_id,),
    )

    rows = cursor.fetchall()
    connection.close()

    messages = []

    for role, content, route, sources_json in rows:

        message = {
            "role": role,
            "content": content,
        }

        if route:
            message["route"] = route

        if sources_json:
            try:
                message["sources"] = json.loads(sources_json)
            except (TypeError, ValueError):
                pass

        messages.append(message)

    return messages


def save_message(chat_id, role, content, route=None, sources=None):
    """Save a message to a specific conversation."""
    connection = sqlite3.connect(CHAT_DB)
    cursor = connection.cursor()

    sources_json = json.dumps(sources) if sources else None

    cursor.execute(
        """
        INSERT INTO messages (chat_id, role, content, route, sources)
        VALUES (?, ?, ?, ?, ?)
        """,
        (chat_id, role, content, route, sources_json),
    )

    connection.commit()
    connection.close()


def update_chat_title(chat_id, title):
    """Update the title of a conversation."""
    connection = sqlite3.connect(CHAT_DB)
    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE chats
        SET title = ?
        WHERE id = ?
        """,
        (title, chat_id),
    )

    connection.commit()
    connection.close()

def repair_old_chat_title():
    """Give the recovered old conversation its actual first-message title."""

    connection = sqlite3.connect(CHAT_DB)
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, content
        FROM messages
        WHERE role = 'user'
        ORDER BY id
        LIMIT 1
        """
    )

    first_message = cursor.fetchone()

    if first_message:

        message_id, content = first_message

        cursor.execute(
            """
            SELECT chat_id
            FROM messages
            WHERE id = ?
            """,
            (message_id,),
        )

        result = cursor.fetchone()

        if result and result[0]:

            chat_id = result[0]

            title = content.strip()

            if len(title) > 40:
                title = title[:40].rstrip() + "..."

            cursor.execute(
                """
                UPDATE chats
                SET title = ?
                WHERE id = ?
                """,
                (title, chat_id),
            )

    connection.commit()
    connection.close()

def clear_chat():
    """Create a new conversation."""
    chat_id = create_chat()

    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []

def delete_chat_history():
    """Delete all saved conversations and messages."""
    connection = sqlite3.connect(CHAT_DB)
    cursor = connection.cursor()

    cursor.execute("DELETE FROM messages")
    cursor.execute("DELETE FROM chats")

    connection.commit()
    connection.close()

    st.session_state.messages = []
    st.session_state.current_chat_id = None
def initialise_session():
    """Initialize NOVA and load the current conversation."""

    init_chat_database()

    migrate_old_messages()

    repair_old_chat_title()

    if "current_chat_id" not in st.session_state:

        chats = get_chats()

        if chats:

            st.session_state.current_chat_id = chats[0][0]

        else:

            st.session_state.current_chat_id = create_chat()

    if "messages" not in st.session_state:

        st.session_state.messages = load_chat_messages(
            st.session_state.current_chat_id
        )


def get_gemini_history():
    """Convert stored messages into Gemini SDK format."""
    return [
        {
            "role": message["role"],
            "parts": [{"text": message["content"]}],
        }
        for message in st.session_state.messages
    ]

def stream_response(api_key, model_name, question):
    """
    Stream an answer from Gemini.

    Uses the same routing (DOCUMENT/MAIL/MEETINGS/WEB/CHAT) as the
    local model via build_routed_prompt() - previously this only
    ever called retrieve_context() (document RAG), so Gemini answers
    never consulted mail or calendar evidence the way Ollama/Groq's
    answers did (e.g. "am I free today?" would get answered from
    Gemini's own guess/web search instead of the user's actual
    calendar). The routed prompt already contains full agent
    instructions and evidence, so it's sent as a single user message
    - build_local_history() supplies the prior-turn context instead
    of the full native message history, matching how the Ollama/Groq
    paths do it.
    """

    request_start = time.perf_counter()

    conversation_history = build_local_history()

    route, sources, prompt = build_routed_prompt(
        question, conversation_history
    )

    log_timing(
        f"[gemini] routing + retrieval took "
        f"{time.perf_counter() - request_start:.2f}s | route={route}"
    )

    client = genai.Client(api_key=api_key)

    generation_start = time.perf_counter()

    response = client.models.generate_content_stream(
        model=model_name,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
            # Only useful for the WEB route now that MAIL/MEETINGS/
            # DOCUMENT questions are grounded via build_routed_prompt()
            # above - live web results still help general/current-
            # events questions the router sends down that path.
            "tools": [{"google_search": {}}],
        },
    )

    first_token = True

    for chunk in response:
        if chunk.text:
            if first_token:
                first_token = False
                log_timing(
                    f"[gemini] time to first token = "
                    f"{time.perf_counter() - generation_start:.2f}s"
                )
            yield chunk.text

    log_timing(
        f"[gemini] full generation took "
        f"{time.perf_counter() - generation_start:.2f}s | "
        f"TOTAL request = {time.perf_counter() - request_start:.2f}s"
    )


def groq_stream_response(api_key, model_name, question):
    """
    Stream an answer from Groq's OpenAI-compatible chat endpoint.

    Uses the same routing (DOCUMENT/MAIL/MEETINGS/WEB/CHAT) as the
    local model via build_routed_prompt() - previously this only
    ever did document RAG, so Groq (and Gemini) answers never
    consulted mail, calendar, or live web search the way the local
    model's answers did. The routed prompt already contains full
    agent instructions and evidence, so it's sent as a single user
    message - no separate system/RAG instruction needed.
    """

    request_start = time.perf_counter()

    conversation_history = build_local_history()

    route, sources, prompt = build_routed_prompt(
        question, conversation_history
    )

    log_timing(
        f"[groq] routing + retrieval took "
        f"{time.perf_counter() - request_start:.2f}s | route={route}"
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    generation_start = time.perf_counter()

    response = GROQ_SESSION.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": messages,
            "stream": True,
        },
        stream=True,
        timeout=60,
    )

    response.raise_for_status()

    first_token = True

    for line in response.iter_lines():

        if not line:
            continue

        decoded = line.decode("utf-8")

        if not decoded.startswith("data: "):
            continue

        payload = decoded[len("data: "):]

        if payload.strip() == "[DONE]":
            break

        try:
            chunk = json.loads(payload)
        except ValueError:
            continue

        delta = (
            chunk.get("choices", [{}])[0]
            .get("delta", {})
            .get("content")
        )

        if delta:

            if first_token:
                first_token = False
                log_timing(
                    f"[groq] time to first token = "
                    f"{time.perf_counter() - generation_start:.2f}s"
                )

            yield delta

    log_timing(
        f"[groq] full generation took "
        f"{time.perf_counter() - generation_start:.2f}s | "
        f"TOTAL request = {time.perf_counter() - request_start:.2f}s"
    )


ROUTER_PROMPT = """You are NOVA's query-routing agent.

Classify the user's question into exactly ONE category:

DOCUMENT
Use DOCUMENT when the user is asking about:
- an uploaded file
- a PDF
- a stored document
- the knowledge base
- information contained in their documents

MAIL
Use MAIL when the user is asking about their email/inbox - a
specific message, who emailed them, whether they received
something, the content of an email, unread messages, etc.

Examples of MAIL questions:
- Did I get an email from Sarah about the budget?
- What did John say in his last email?
- Do I have any unread emails from the finance team?
- Find the email with the flight itinerary.

MEETINGS
Use MEETINGS when the user is asking about their calendar,
schedule, or meetings - upcoming events, when something is
scheduled, who's attending, availability, etc.

Examples of MEETINGS questions:
- What's on my calendar tomorrow?
- When is my next meeting with the design team?
- Do I have anything scheduled this afternoon?
- Am I free on Friday?

SEND_MAIL
Use SEND_MAIL when the user wants NOVA to compose and send an
email - an imperative action, not a question about existing mail.

Examples of SEND_MAIL requests:
- Send an email to sarah@company.com telling her the report is done.
- Email John and ask him to review the doc.
- Compose an email to the finance team about the budget delay.
- Reply to Amit saying I'll be there at 3pm.

SCHEDULE_MEETING
Use SCHEDULE_MEETING when the user wants NOVA to add a new event
to the calendar - an imperative action, not a question about
existing events.

Examples of SCHEDULE_MEETING requests:
- Schedule a meeting with the design team tomorrow at 2pm.
- Book a call with Priya on Friday morning.
- Add "dentist appointment" to my calendar next Tuesday at 10am.
- Set up a 30 minute sync with the finance team this afternoon.

LEAVE_REQUEST
Use LEAVE_REQUEST when the user wants NOVA to apply for time off/
leave, or asks about their own leave balance or leave history - an
imperative action or a question about their own leave, not a
question about someone else's schedule.

Examples of LEAVE_REQUEST requests:
- Apply for sick leave tomorrow.
- I want to take annual leave from Aug 25 to Aug 28.
- Book casual leave for Friday.
- How many sick leave days do I have left?
- What's my leave balance?

PO_REQUEST
Use PO_REQUEST when the user wants NOVA to raise/create a Purchase
Order, or asks about the status/history of a PO they raised - an
imperative action or a question about their own PO(s), not a
general question about purchasing in the abstract.

Examples of PO_REQUEST requests:
- Raise a PO for 50 laptops from Dell at $1200 each.
- Create a purchase order for office chairs from IKEA, 10 units at $150.
- Submit a PO to Acme Supplies for 200 units of widget-A at $12.50, marketing department.
- What's the status of my last PO?
- Show me my purchase order history.

WEB
Use WEB when the user needs a SPECIFIC, real-world fact that can
change over time - a person's current role, a company, an event,
a date, a price, breaking news, etc.

Examples of WEB questions:
- Who is the wife of Virat Kohli?
- Who is the current CEO of Microsoft?
- When was Apple founded?
- What is the latest news about NVIDIA?
- What is the current price of Bitcoin?
- What is the weather today?
- Who won the 2026 World Cup?

CHAT
Use CHAT for:
- casual conversation, greetings, opinions, creative writing
- TIMELESS/CONCEPTUAL questions - definitions, explanations of
  how something works, general knowledge that doesn't change.
  Answer these from your own knowledge; they do NOT need a web
  search.

Examples of CHAT questions (answer directly, don't say WEB):
- What is a GPU?
- What is machine learning?
- How does photosynthesis work?
- What is the tallest mountain in the world?
- Explain how a car engine works.

IMPORTANT:
A factual question about a person, company, event, date,
relationship, price, or anything that can change should normally
be WEB. A question asking what a general concept/term MEANS
should be CHAT - the model already knows it and it isn't going to
change. Questions about the user's OWN email or calendar are
never WEB - those are MAIL or MEETINGS respectively, even if they
mention a company, date, or person's name. A request to compose
and send mail, or to add something to the calendar, is SEND_MAIL
or SCHEDULE_MEETING respectively - NOT MAIL or MEETINGS, which are
only for questions about what already exists. The distinguishing
signal is imperative action verbs ("send", "email [someone]",
"reply to", "schedule", "book", "add to my calendar") versus
questions ("did I get", "what's on my calendar"). A request to
apply for time off, or a question about the user's own leave
balance/history, is LEAVE_REQUEST. A request to raise/create a
Purchase Order, or a question about the user's own PO status/
history, is PO_REQUEST.
A short follow-up (e.g. "and her sister?", "what about in 2020?") inherits
the category of the RECENT CONVERSATION below if it's asking to extend
the same topic.

RECENT CONVERSATION (for context only, may be empty):
{history}

Reply with ONE WORD ONLY:
DOCUMENT
MAIL
MEETINGS
SEND_MAIL
SCHEDULE_MEETING
LEAVE_REQUEST
PO_REQUEST
WEB
CHAT

Question:
{question}

Label:"""


# ---------------------------------------------------
# Small-talk / greeting phrases - shared between:
#   - route_query()'s fast CHAT path (below)
#   - the follow-up-question skip check (see main(), near
#     followup_executor.submit) - a bare greeting has no real
#     content to build a follow-up question on, so we don't even
#     ask the model for one.
# ---------------------------------------------------
GREETING_SIGNALS = {
    "hi", "hii", "hiya", "hello", "hey", "hey there", "yo", "sup",
    "what's up", "whats up", "wassup",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how are you doing", "how r u", "how are u",
    "how's it going", "hows it going",
    "thanks", "thank you", "thanks a lot", "thank you so much",
    "ok", "okay", "cool", "nice", "great", "lol",
    "bye", "goodbye", "see you", "see ya",
}

# ---------------------------------------------------
# Shown in place of real follow-ups whenever there's no actual
# content to generate one FROM - a bare greeting (skipped before
# even calling the model - see is_bare_greeting in main()), or a
# turn where the model's suggestions were all generic filler and
# got filtered out by generate_followup_questions(). Rather than
# leave the "you might also ask" row empty, offer generic task
# starters that showcase what NOVA can actually do.
# ---------------------------------------------------
STARTER_SUGGESTIONS = [
    "Make me a study plan",
    "Help me solve a math problem",
    "Help me think through a situation I'm dealing with",
    "Summarize a document for me",
    "Draft an email for me",
    "What's on my calendar today?",
    "Explain a concept I'm stuck on",
    "Help me plan my week",
    "Search the web for something",
    "Quiz me on a topic",
]


def route_query(question, conversation_history=""):
    """
    Hybrid query-routing agent.

    Fast path:
        Obvious DOCUMENT / WEB queries are classified immediately.

    Fallback:
        Ambiguous queries are sent to the Ollama routing agent,
        along with a little conversation history so follow-ups
        ("what about her sister?") can still be classified
        correctly even though they have no keywords of their own.

    Returns:
        DOCUMENT
        MAIL
        MEETINGS
        WEB
        CHAT
    """

    q = question.lower().strip()

    # =========================================================
    # FAST DOCUMENT ROUTING
    # =========================================================

    document_signals = [
        "uploaded document",
        "uploaded file",
        "uploaded pdf",
        "uploaded files",
        "according to the document",
        "according to the file",
        "according to my document",
        "according to my file",
        "in the document",
        "in the file",
        "from the document",
        "from the file",
        "my document",
        "my file",
        "my pdf",
        "knowledge base",
        "stored document",
        "stored file",
    ]

    if any(signal in q for signal in document_signals):
        log_timing("route_query -> 'DOCUMENT' (fast)")
        return "DOCUMENT"

    # =========================================================
    # FAST SELF-INFO ROUTING
    #
    # Meta-questions about NOVA ITSELF ("what embedding model does
    # nova use", "what LLM are you running", "what model powers
    # this") aren't answerable by any real agent - WEB searches the
    # live internet and (correctly) finds nothing, since this is a
    # local app; DOCUMENT/MAIL/MEETINGS have nothing to do with it
    # either. Before this route existed, these questions fell
    # through to WEB (matching "current"/generic wh- patterns) and
    # came back with a "couldn't find reliable information" dead
    # end. Answer directly from the real constants instead.
    # =========================================================

    self_info_signals = [
        "embedding model", "what model does nova use",
        "what model do you use", "what llm", "which llm",
        "what model are you running", "which model are you running",
        "what model powers", "router model", "answer model",
        "what ai model", "which ai model",
    ]

    if any(signal in q for signal in self_info_signals):
        log_timing("route_query -> 'SELF_INFO' (fast)")
        return "SELF_INFO"

    # =========================================================
    # FAST CHAT ROUTING (greetings / social small talk)
    #
    # Bare small talk like "hi" has no topical keyword at all, so
    # without this it falls all the way through to the LLM router
    # fallback below - where a small 1.5B model can plausibly (and
    # wrongly) label it WEB, since the router prompt has no example
    # that short. That sends it out to a live web search and back
    # with something like a dictionary definition instead of just
    # responding to the greeting.
    #
    # Matched against the ENTIRE (punctuation-stripped) message, not
    # a substring - so this never swallows a real question that
    # happens to start with a greeting ("hi, what's Apple's stock
    # price today?" still falls through to WEB normally).
    # =========================================================

    q_exact = q.strip(" !.,?~-")

    if q_exact in GREETING_SIGNALS:
        log_timing("route_query -> 'CHAT' (fast, greeting)")
        return "CHAT"

    # Self-introductions ("im kishore", "I'm Kishore", "my name is
    # Kishore", "this is Kishore") are social small talk, not a
    # request for information ABOUT that person - but they contain
    # a bare name with no other context, which the small router
    # model tends to misread as "who is <name>" and mislabel WEB
    # (sending it out to search and coming back with "couldn't
    # find reliable information"). Catch the shape before that can
    # happen. Anchored at the start and requires the message to be
    # just the intro (optionally trailing punctuation) - a longer
    # message that happens to start this way ("I'm Kishore, can you
    # check my email?") still falls through normally.
    if re.match(
        r"^(i'?m|i am|my name is|this is|call me)\s+[a-z][a-z .'\-]{0,40}[!.]*$",
        q_exact,
    ):
        log_timing("route_query -> 'CHAT' (fast, self-introduction)")
        return "CHAT"

    # =========================================================
    # FAST SEND-MAIL ROUTING
    #
    # Checked BEFORE the read-only MAIL signals below - a send
    # request ("send an email to Sarah...") also contains "email"
    # and would otherwise be misrouted into the inbox-search agent.
    # =========================================================

    send_mail_signals = [
        "send an email", "send email", "send an e-mail",
        "send a mail", "send mail to", "compose an email",
        "compose email", "compose an e-mail", "write an email to",
        "draft an email to", "draft me an email", "reply to",
        "forward this to", "forward that to",
    ]

    if any(signal in q for signal in send_mail_signals):
        log_timing("route_query -> 'SEND_MAIL' (fast)")
        return "SEND_MAIL"

    # "reply to"/"forward ... to" as a literal substring misses real
    # phrasing like "reply thanks to Kishore" or "reply thanks in
    # mail to Kishore" - the verb and "to" aren't adjacent. Allow up
    # to 6 filler words between them. Anchored at the start of the
    # message, since a command ("reply ...") and a question about
    # past mail ("did I get a reply from...") don't share that
    # shape - the latter doesn't open with the bare verb.
    if re.match(r"^(reply|forward)\b(?:\s+\S+){0,6}\s+to\b", q):
        log_timing("route_query -> 'SEND_MAIL' (fast, verb...to)")
        return "SEND_MAIL"

    # "send hi to Kishore M S in mail" - a bare "send ... to ..."
    # command whose only mail keyword ("mail"/"email") shows up
    # *after* the recipient, not adjacent to "send". None of the
    # literal send_mail_signals phrases match this shape, so
    # without this it fell all the way through to the LLM router,
    # which sometimes mislabels it as MAIL (read-only inbox search)
    # instead of SEND_MAIL. Anchored at the start (a command opens
    # with the verb) and requires a mail keyword somewhere in the
    # message to avoid catching unrelated "send X to Y" requests.
    if re.match(r"^send\b(?:\s+\S+){0,10}\s+to\b", q) and re.search(
        r"\b(mail|email|e-mail)\b", q
    ):
        log_timing("route_query -> 'SEND_MAIL' (fast, send...to...mail)")
        return "SEND_MAIL"

    # =========================================================
    # FAST SCHEDULE-MEETING ROUTING
    #
    # Checked BEFORE the read-only MEETINGS signals below, for the
    # same reason - "schedule a meeting" contains "meeting" and
    # "schedule".
    # =========================================================

    schedule_meeting_signals = [
        "schedule a meeting", "schedule meeting", "schedule a call",
        "set up a meeting", "set up a call", "book a meeting",
        "book a call", "arrange a meeting", "arrange a call",
        "add to my calendar", "add this to my calendar",
        "add an event", "create a meeting", "create an event",
        "schedule an event",
    ]

    if any(signal in q for signal in schedule_meeting_signals):
        log_timing("route_query -> 'SCHEDULE_MEETING' (fast)")
        return "SCHEDULE_MEETING"

    # The literal phrases above miss natural filler like "schedule
    # ME a meeting with X" or "book a call WITH PRIYA FRIDAY" (verb
    # and noun aren't adjacent), and miss nouns like "appointment"
    # or "sync" entirely - which would otherwise fall through to
    # the read-only MEETINGS signals below ("appointment" is in
    # that list) and get misrouted as a lookup. Anchored at the
    # start, same reasoning as the SEND_MAIL regex above: a command
    # opens with the verb, a question about existing events doesn't.
    if re.match(
        r"^(please\s+)?(schedule|book|set up|arrange|create|add|put)\b"
        r"(?:\s+\S+){0,6}\s+"
        r"(meeting|call|event|appointment|sync|catch-?up|reminder|calendar)\b",
        q,
    ):
        log_timing("route_query -> 'SCHEDULE_MEETING' (fast, verb...noun)")
        return "SCHEDULE_MEETING"

    # =========================================================
    # FAST LEAVE-REQUEST ROUTING
    #
    # Checked before MEETINGS below - "leave" alone is ambiguous,
    # but these phrases are specific enough not to collide with
    # calendar/meeting language.
    # =========================================================

    leave_signals = [
        "apply for leave", "apply leave", "request leave",
        "take leave", "book leave", "leave request",
        "sick leave", "annual leave", "casual leave", "vacation leave",
        "leave balance", "leave history", "days of leave left",
        "how many leave days",
    ]

    if any(signal in q for signal in leave_signals):
        log_timing("route_query -> 'LEAVE_REQUEST' (fast)")
        return "LEAVE_REQUEST"

    if re.match(
        r"^(please\s+)?(apply|request|take|book)\b(?:\s+\S+){0,4}\s+leave\b",
        q,
    ):
        log_timing("route_query -> 'LEAVE_REQUEST' (fast, verb...leave)")
        return "LEAVE_REQUEST"

    # =========================================================
    # FAST PO-REQUEST ROUTING
    #
    # "PO" alone is too short/ambiguous to substring-match safely,
    # so it's only matched as a whole word below; the phrase list
    # covers the common spelled-out forms.
    # =========================================================

    po_signals = [
        "purchase order", "purchase orders", "raise a po", "raise po",
        "create a po", "create po", "submit a po", "submit po",
        "po request", "po status", "my po", "my pos",
    ]

    if any(signal in q for signal in po_signals):
        log_timing("route_query -> 'PO_REQUEST' (fast)")
        return "PO_REQUEST"

    if re.search(r"\bpo\b", q) and re.search(
        r"\b(raise|create|submit|make|status|history|approve[d]?|pending)\b", q
    ):
        log_timing("route_query -> 'PO_REQUEST' (fast, bare 'po' + verb)")
        return "PO_REQUEST"

    if re.match(
        r"^(please\s+)?(raise|create|submit|make|place)\b(?:\s+\S+){0,4}\s+"
        r"(purchase order|po)\b",
        q,
    ):
        log_timing("route_query -> 'PO_REQUEST' (fast, verb...po)")
        return "PO_REQUEST"

    # =========================================================
    # FAST MAIL ROUTING
    # =========================================================

    mail_signals = [
        "email", "e-mail", "emails", "e-mails", "inbox", "mailbox",
        "unread mail", "unread email", "unread emails",
        "who emailed", "emailed me", "mailed me", "sent me an email",
        "sent me a mail",
    ]

    if any(signal in q for signal in mail_signals):
        log_timing("route_query -> 'MAIL' (fast)")
        return "MAIL"

    # Bare "mail"/"mails" ("summarize the last received mail",
    # "any new mail?") wasn't covered by mail_signals above - those
    # only match "email"/"e-mail" or specific compound phrases, so
    # a query using the word "mail" on its own fell through to the
    # LLM router fallback, which has been observed to mislabel it
    # as WEB. Word-boundary matched (not a plain substring) so this
    # doesn't fire on words like "mailbox" (already handled above)
    # or "blackmail"/"chainmail".
    if re.search(r"\bmails?\b", q):
        log_timing("route_query -> 'MAIL' (fast, bare 'mail')")
        return "MAIL"

    # =========================================================
    # FAST MEETINGS ROUTING
    # =========================================================

    meeting_signals = [
        "meeting", "meetings", "calendar", "schedule", "scheduled",
        "appointment", "appointments", "agenda",
        "am i free", "am i busy", "what's on my calendar",
        "whats on my calendar",
    ]

    if any(signal in q for signal in meeting_signals):
        log_timing("route_query -> 'MEETINGS' (fast)")
        return "MEETINGS"

    # =========================================================
    # FAST WEB ROUTING
    #
    # Word-boundary matching (\b...\b) rather than raw substrings -
    # "born" as a plain substring matches inside "airborne" or
    # "reborn" and misroutes unrelated questions into the web
    # agent. Multi-word phrases keep matching as phrases.
    # =========================================================

    web_signal_words = [
        "who", "wife", "husband", "father", "mother", "son",
        "daughter", "brother", "sister", "born", "died", "founded",
        "latest", "today", "current", "recent", "news", "price",
        "prices", "weather", "stock", "stocks", "trending", "2026",
    ]

    web_signal_phrases = [
        "who is ", "who was ", "who are ",
        "wife of ", "husband of ", "father of ", "mother of ",
        "son of ", "daughter of ", "brother of ", "sister of ",
        "founder of ", "ceo of ", "president of ",
        "capital of ", "population of ",
        "exchange rate", "right now", "this week", "this month",
    ]

    if any(re.search(rf"\b{re.escape(word)}\b", q) for word in web_signal_words):
        log_timing("route_query -> 'WEB' (fast, word)")
        return "WEB"

    if any(phrase in q for phrase in web_signal_phrases):
        log_timing("route_query -> 'WEB' (fast, phrase)")
        return "WEB"

    # =========================================================
    # FAST FOLLOW-UP ROUTING
    #
    # A short follow-up with no topical keyword of its own ("from
    # kishore", "and tomorrow?", "what about the design team?")
    # matches none of the signal lists above and falls through to
    # the LLM router fallback below. That fallback IS given the
    # conversation history and told to inherit the prior turn's
    # category (see ROUTER_PROMPT), but in practice a 1.5B model
    # at num_predict=10 is not reliable at that kind of contextual
    # inference - e.g. "from kishore" right after a MAIL turn has
    # been observed to come back DOCUMENT instead of MAIL.
    #
    # Short-circuit that: if the previous turn was routed to a
    # read-style agent and this message is short (no room for it
    # to plausibly be introducing a whole new, unrelated topic),
    # just inherit that route directly instead of gambling on the
    # small model. SEND_MAIL/SCHEDULE_MEETING are excluded - those
    # are one-off actions, not something a vague follow-up should
    # silently repeat.
    # =========================================================

    previous_route = st.session_state.get("last_route")

    # A genuine follow-up fragment ("from kishore", "and tomorrow?",
    # "what about the design team?") has no subject/verb of its own
    # - it only makes sense attached to the previous turn. But word
    # count alone doesn't distinguish that from a short, but fully
    # self-contained, NEW question ("what embedding model does nova
    # use?") - that has its own subject ("nova") and verb ("does
    # ... use"), so it was being wrongly forced into whatever agent
    # handled the previous turn just for being <= 6 words. Skip the
    # inherit when the message is a complete WH-question (a wh-word
    # followed by an auxiliary/verb) UNLESS it also uses a
    # referential pronoun ("that", "it", "this"...) pointing back at
    # the previous answer - "how does THAT work" is still a real
    # follow-up, "what embedding model does NOVA use" is not.
    self_contained_question_re = re.compile(
        r"^(what|why|how|when|where|which|who)\b.{0,60}\b"
        r"(is|are|was|were|do|does|did|can|will|would|should)\b"
    )
    referential_re = re.compile(r"\b(it|that|this|these|those|they|them)\b")

    looks_like_new_topic = (
        self_contained_question_re.match(q) and not referential_re.search(q)
    )

    if (
        previous_route in ("MAIL", "MEETINGS", "DOCUMENT", "WEB")
        and len(q.split()) <= 6
        and not looks_like_new_topic
    ):
        log_timing(
            f"route_query -> {previous_route!r} "
            "(fast, inherited from previous turn)"
        )
        return previous_route

    # =========================================================
    # LLM ROUTER FALLBACK
    # =========================================================

    t0 = time.perf_counter()

    router_prompt = ROUTER_PROMPT.format(
        history=conversation_history or "(none)",
        question=question,
    )

    try:
        response = OLLAMA_SESSION.post(
            OLLAMA_URL,
            json={
                "model": ROUTER_MODEL,
                "prompt": router_prompt,
                "stream": False,
                "keep_alive": "24h",
                "options": {
                    "temperature": 0.0,
                    "num_ctx": LOCAL_MODEL_NUM_CTX,
                    # 3 tokens was cutting "DOCUMENT" off mid-word,
                    # so neither check matched and it silently fell
                    # through to ungrounded CHAT. 10 gives the
                    # longest label ("MEETINGS") room to complete.
                    "num_predict": 10,
                    "num_gpu": 99,
                },
            },
            timeout=15,
        )

        response.raise_for_status()

        label = (
            response.json()
            .get("response", "")
            .strip()
            .upper()
        )

    except Exception as error:

        log_timing(
            f"route_query FAILED after "
            f"{time.perf_counter() - t0:.2f}s: {error}"
        )

        return "CHAT"

    log_timing(
        f"route_query -> {label!r} in "
        f"{time.perf_counter() - t0:.2f}s"
    )

    if "DOCUMENT" in label:
        return "DOCUMENT"

    # Checked before the plain MAIL/MEETING checks below, since
    # "SEND_MAIL" and "SCHEDULE_MEETING" both contain those
    # substrings.
    if "SEND_MAIL" in label or "SEND MAIL" in label:
        return "SEND_MAIL"

    if "SCHEDULE" in label:
        return "SCHEDULE_MEETING"

    if "LEAVE" in label:
        return "LEAVE_REQUEST"

    if "PO_REQUEST" in label or "PO REQUEST" in label:
        return "PO_REQUEST"

    if "MAIL" in label:
        return "MAIL"

    if "MEETING" in label:
        return "MEETINGS"

    if "WEB" in label:
        return "WEB"

    return "CHAT"

def build_local_history(max_turns=3, max_chars_per_message=250):
    """
    Short conversation-history block for the local model.

    stream_ollama_response() uses Ollama's raw /api/generate
    endpoint (not a chat endpoint), so - unlike Gemini, which gets
    the full history via get_gemini_history() - it never saw prior
    turns at all. Each question was answered in isolation. This
    reconstructs a small, capped slice of recent turns so the
    model has continuity without blowing the (still limited)
    context budget.
    """

    messages = st.session_state.get("messages", [])

    # The current question was already appended to messages before
    # this is called - exclude it, we only want PRIOR turns.
    prior_messages = messages[:-1] if messages else []

    recent_messages = prior_messages[-(max_turns * 2):]

    lines = []

    for message in recent_messages:

        speaker = "User" if message["role"] == "user" else "NOVA"
        content = message["content"][:max_chars_per_message]

        lines.append(f"{speaker}: {content}")

    return "\n".join(lines)


def _parse_json_object(raw_text):
    """
    Best-effort JSON extraction from a local model's raw response -
    small local models sometimes wrap the JSON in a sentence or a
    markdown fence despite being told to reply with only JSON.
    """

    raw_text = (raw_text or "").strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def _call_extraction_model(extraction_prompt, num_predict=300):
    """
    Shared Ollama call for the field-extraction agents below - same
    model/session as the router, just a longer, structured prompt.
    """

    response = OLLAMA_SESSION.post(
        OLLAMA_URL,
        json={
            "model": ROUTER_MODEL,
            "prompt": extraction_prompt,
            "stream": False,
            "keep_alive": "24h",
            "options": {
                "temperature": 0.0,
                "num_ctx": LOCAL_MODEL_NUM_CTX,
                "num_predict": num_predict,
                "num_gpu": 99,
            },
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.json().get("response", "")


# ================================
# NAME -> EMAIL ADDRESS RESOLUTION
#
# The extraction model is told to pass through a bare name (e.g.
# "Kishore M S") when the user doesn't spell out an address. Rather
# than hard-failing there, search the mailbox for that name and
# pull a real address out of the matching messages' headers - this
# is what "reply thanks to Kishore" should mean in an app that
# already has Kishore's past emails on hand.
# ================================

EMAIL_ADDRESS_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# search_mail() formats each match as a "FROM: <sender>" line inside
# a larger block that also includes the message BODY. Only this
# line is trustworthy as "this is the sender's own address" - the
# body can and does contain other people's addresses (signatures,
# quoted threads, mailing footers), and picking those up gives a
# wrong-but-real-looking address instead of a clear failure.
FROM_HEADER_RE = re.compile(r"^FROM:\s*(.+)$", re.MULTILINE)


def _resolve_one_recipient(name):
    """
    Looks up a single bare name against the mailbox and returns the
    most likely email address, or None if nothing usable was found.
    Only ever trusts the sender field of matched messages - never
    text found inside a message body.
    """

    if "@" in name:
        return name.strip()

    try:
        context, _sources = search_mail(
            name, max_results=5, require_keyword_match=True
        )
    except Exception as error:
        log_timing(f"_resolve_one_recipient FAILED for '{name}': {error}")
        return None

    if not context:
        return None

    # One address per matched email's FROM line - each line may be
    # "Kishore M S <kishore.real@gmail.com>" or a bare address.
    found = []
    for from_line in FROM_HEADER_RE.findall(context):
        match = EMAIL_ADDRESS_RE.search(from_line)
        if match:
            found.append(match.group(0))

    if not found:
        return None

    # Most frequent sender wins - if several matched emails came
    # from the same address, that's a stronger signal than whichever
    # was scanned first.
    counts = {}
    for address in found:
        counts[address.lower()] = counts.get(address.lower(), 0) + 1

    return max(counts, key=counts.get)


def _resolve_recipient_list(raw_value):
    """
    Resolves a comma-separated list of names/addresses, leaving
    real addresses untouched.

    Returns:
        (resolved: str, unresolved: list[str]) - `resolved` is the
        comma-joined list of addresses actually found; `unresolved`
        holds any names that couldn't be matched to an address.
    """

    names = [part.strip() for part in raw_value.split(",") if part.strip()]

    resolved = []
    unresolved = []

    for name in names:
        address = _resolve_one_recipient(name)
        if address:
            resolved.append(address)
        else:
            unresolved.append(name)

    return ", ".join(resolved), unresolved


def _extract_addresses(name_field, email_field, question):
    """
    Resolves a recipient (or cc) list, given the extraction model's
    separate name/email guesses for it.

    An address the model supplied is only trusted if it appears
    verbatim (case-insensitive) somewhere in the user's own
    message - otherwise it's treated as a guess, not something the
    user actually typed, and resolution falls back to searching the
    mailbox by name instead. This is what stops a small model's
    plausible-looking but fabricated address (e.g. turning "Kishore
    M S" into "kishorems@company.com") from silently ending up in
    the draft.

    Returns:
        (resolved: str, unresolved: list[str]) - same shape as
        _resolve_recipient_list.
    """

    names = [part.strip() for part in name_field.split(",") if part.strip()]
    emails = [part.strip() for part in email_field.split(",") if part.strip()]

    if names and emails and len(names) == len(emails):
        pairs = list(zip(names, emails))
    elif emails and not names:
        pairs = [("", email) for email in emails]
    elif names:
        pairs = [(name, "") for name in names]
    else:
        pairs = []

    question_lower = question.lower()
    resolved = []
    unresolved = []

    for name, email in pairs:

        if email and email.lower() in question_lower:
            resolved.append(email)
            continue

        lookup = name or email

        if not lookup:
            continue

        address = _resolve_one_recipient(lookup)
        if address:
            resolved.append(address)
        else:
            unresolved.append(lookup)

    return ", ".join(resolved), unresolved


def extract_mail_fields(question):
    """
    Pulls structured send-mail fields (to/cc/subject/body) out of a
    natural-language SEND_MAIL request.

    Returns:
        (fields, error) - exactly one of these is set. `fields` is
        a dict whose keys match send_mail()'s parameters, ready to
        call as send_mail(**fields). `error` is a short, user-
        facing message explaining what's missing or what went
        wrong.
    """

    extraction_prompt = f"""
Extract the fields needed to send an email from the request below.
Reply with ONLY a JSON object, no other text, in exactly this shape:

{{"to_name": "...", "to_email": "...", "cc_name": "", "cc_email": "", "bcc_name": "", "bcc_email": "", "subject": "...", "body": "..."}}

- "to_name": the name or identifier the user used for the
  recipient (e.g. "Kishore M S"), or "" if they only gave an
  address.
- "to_email": the recipient's email address, ONLY if the user
  literally typed it out, character-for-character, in the request
  below. If the user only gave a name, leave this "". NEVER invent,
  guess, or auto-complete an address (e.g. do not turn a name into
  "name@company.com" or any other made-up domain) - an incorrect
  guess is worse than leaving this blank.
- "cc_name" / "cc_email": same rules, for a cc'd person - both ""
  if there's no cc.
- "bcc_name" / "bcc_email": same rules, for a bcc'd person (a
  recipient the user wants hidden from everyone else, e.g. "bcc my
  manager") - both "" if there's no bcc.
- "subject": a short subject line. If the user didn't give one,
  write a short one that fits the body.
- "body": the email body, written out in full sentences, matching
  what the user wants said. If the user gave exact wording, use it.

REQUEST:
{question}

JSON:""".strip()

    try:
        raw = _call_extraction_model(extraction_prompt)
    except Exception as error:
        log_timing(f"extract_mail_fields FAILED: {error}")
        return None, f"Couldn't reach the local model to read that request: {error}"

    parsed = _parse_json_object(raw)

    if not parsed:
        return None, "I couldn't parse the details of that email request."

    to_name = str(parsed.get("to_name", "")).strip()
    to_email = str(parsed.get("to_email", "")).strip()
    cc_name = str(parsed.get("cc_name", "")).strip()
    cc_email = str(parsed.get("cc_email", "")).strip()
    bcc_name = str(parsed.get("bcc_name", "")).strip()
    bcc_email = str(parsed.get("bcc_email", "")).strip()
    subject = str(parsed.get("subject", "")).strip()
    body = str(parsed.get("body", "")).strip()

    to, unresolved = _extract_addresses(to_name, to_email, question)

    if unresolved:
        return None, (
            "I couldn't find an email address for "
            f"{', '.join(unresolved)} in your inbox - could you give "
            "me the address directly?"
        )

    cc = ""
    if cc_name or cc_email:
        cc, cc_unresolved = _extract_addresses(cc_name, cc_email, question)
        if cc_unresolved:
            return None, (
                "I couldn't find an email address for "
                f"{', '.join(cc_unresolved)} (cc) in your inbox - could "
                "you give me the address directly?"
            )

    bcc = ""
    if bcc_name or bcc_email:
        bcc, bcc_unresolved = _extract_addresses(bcc_name, bcc_email, question)
        if bcc_unresolved:
            return None, (
                "I couldn't find an email address for "
                f"{', '.join(bcc_unresolved)} (bcc) in your inbox - could "
                "you give me the address directly?"
            )

    if not to:
        return None, "I couldn't tell who to send that to - could you give me an email address?"

    if not body:
        return None, "I couldn't tell what the email should say - could you give me the message?"

    return {
        "to": to,
        "subject": subject or "(no subject)",
        "body": body,
        "cc": cc or None,
        "bcc": bcc or None,
    }, None


def extract_meeting_fields(question):
    """
    Pulls structured scheduling fields (title/start/end/location/
    attendees) out of a natural-language SCHEDULE_MEETING request.

    Returns:
        (fields, error) - exactly one of these is set. `fields` is
        a dict whose keys match schedule_meeting()'s parameters,
        ready to call as schedule_meeting(**fields). `error` is a
        short, user-facing message.
    """

    now = datetime.now()
    today_line = now.strftime("%A, %Y-%m-%d %H:%M")

    extraction_prompt = f"""
Extract the fields needed to schedule a calendar event from the
request below. Reply with ONLY a JSON object, no other text, in
exactly this shape:

{{"title": "...", "date": "YYYY-MM-DD", "time": "HH:MM", "duration_minutes": 30, "location": "", "attendees": ""}}

CURRENT DATE/TIME: {today_line}
Resolve relative dates ("tomorrow", "next Tuesday", "Friday
afternoon") against this. Use 24-hour "HH:MM" time. If no time is
given, use "09:00". If no duration is given, use 30. "location"
and "attendees" (comma-separated names/emails) are "" if not
mentioned.

REQUEST:
{question}

JSON:""".strip()

    try:
        raw = _call_extraction_model(extraction_prompt, num_predict=200)
    except Exception as error:
        log_timing(f"extract_meeting_fields FAILED: {error}")
        return None, f"Couldn't reach the local model to read that request: {error}"

    parsed = _parse_json_object(raw)

    if not parsed:
        return None, "I couldn't parse the details of that meeting request."

    title = str(parsed.get("title", "")).strip() or "Meeting"
    date_str = str(parsed.get("date", "")).strip()
    time_str = str(parsed.get("time", "")).strip() or "09:00"
    location = str(parsed.get("location", "")).strip()
    attendees = str(parsed.get("attendees", "")).strip()

    try:
        duration = int(parsed.get("duration_minutes", 30))
    except (TypeError, ValueError):
        duration = 30

    try:
        start = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None, "I couldn't work out the date/time for that meeting - could you give an explicit date and time?"

    end = start + timedelta(minutes=duration)

    return {
        "title": title,
        "start": start,
        "end": end,
        "location": location,
        "description": "",
        "attendees": attendees or None,
        # A literal time the user gave ("at 3pm today") can still
        # resolve to a moment that's already gone by the time this
        # runs - e.g. asking for "3pm today" at 3:47pm. Nothing
        # upstream catches that; flag it here so the confirmation
        # card can warn before the event gets silently written to
        # the past instead of quietly going through.
        "in_past": start < datetime.now(),
    }, None


def extract_leave_fields(question):
    """
    Pulls structured leave-request fields (leave_type/start/end/
    reason) out of a natural-language LEAVE_REQUEST request.

    Returns:
        (fields, error) - exactly one of these is set. `fields` has
        the keys apply_leave() expects (user/leave_type/start_date/
        end_date/reason), plus the validation results (ok/errors/
        warnings/info) so the confirmation card can show them
        without a second round-trip.
    """

    now = datetime.now()
    today_line = now.strftime("%A, %Y-%m-%d")

    extraction_prompt = f"""
Extract the fields needed to apply for leave from the request below.
Reply with ONLY a JSON object, no other text, in exactly this shape:

{{"leave_type": "annual", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "reason": ""}}

CURRENT DATE: {today_line}
Resolve relative dates ("tomorrow", "next Monday", "Aug 25 to Aug
28") against this. "leave_type" must be one of: annual, sick,
casual - pick the closest match, defaulting to "annual" if the
request doesn't say. If only one date is mentioned, use it for both
start_date and end_date. "reason" is a short free-text reason if the
user gave one, else "".

REQUEST:
{question}

JSON:""".strip()

    try:
        raw = _call_extraction_model(extraction_prompt, num_predict=150)
    except Exception as error:
        log_timing(f"extract_leave_fields FAILED: {error}")
        return None, f"Couldn't reach the local model to read that request: {error}"

    parsed = _parse_json_object(raw)

    if not parsed:
        return None, "I couldn't parse the details of that leave request."

    leave_type = str(parsed.get("leave_type", "")).strip().lower() or "annual"
    if leave_type not in ("annual", "sick", "casual"):
        leave_type = "annual"

    start_str = str(parsed.get("start_date", "")).strip()
    end_str = str(parsed.get("end_date", "")).strip() or start_str
    reason = str(parsed.get("reason", "")).strip()

    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError:
        return None, "I couldn't work out the date(s) for that leave request - could you give an explicit date or range?"

    # "me" - the primary configured user, same convention as
    # schedule_meeting()/check_group_availability() default. NOVA is
    # single-tenant per deployment; the person chatting IS "me".
    user = "me"

    ok, errors, warnings, info = validate_leave_request(
        user, leave_type, start_date, end_date, reason
    )

    return {
        "user": user,
        "leave_type": leave_type,
        "start_date": start_date,
        "end_date": end_date,
        "reason": reason,
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }, None


_CURRENCY_WORD = r"(?:rupees|rupee|rs\.?|inr|₹|dollars|dollar|usd|\$)"
DEFAULT_VENDOR_EMAIL = os.environ.get(
    "NOVA_PO_DEFAULT_VENDOR_EMAIL", "mskishore.studies@gmail.com"
)
_STATED_TOTAL_RE = re.compile(
    r"(?:" + _CURRENCY_WORD + r")\s*(\d[\d,]*(?:\.\d+)?)\b"
    r"(?!\s*(?:each|per|/|a piece|apiece))"
    r"|(\d[\d,]*(?:\.\d+)?)\s*" + _CURRENCY_WORD + r"\b"
    r"(?!\s*(?:each|per|/|a piece|apiece))"
    # Bare "for <number>" with NO currency word at all, e.g. "...to
    # reliance digital for 400000". Only counts as a total when the
    # number isn't immediately followed by another word - that
    # excludes cases like "for 10 laptops" (quantity, not total) and
    # "for 50 each"/"...per unit" (already excluded by the other two
    # alternatives' lookaheads, but "each"/"per" also start with a
    # letter so this lookahead catches them too).
    r"|\bfor\s+(\d[\d,]*(?:\.\d+)?)\b(?!\s*[a-zA-Z])",
    re.IGNORECASE,
)


def _extract_stated_total(text):
    """
    Best-effort pull of a single currency amount the user typed
    directly as a TOTAL (e.g. "for 30,000 rupees", "₹30000", "$150",
    or a bare "...for 400000" with no currency word at all), used to
    sanity-check the LLM-extracted PO total. Explicitly skips amounts
    followed by "each"/"per"/etc, since those are per-unit prices,
    not totals (e.g. "100 notebooks ... for 50 rupees each" is
    unit_price=50, not total=50). Returns None if zero or more than
    one distinct amount is found, since with several numbers in play
    we can't safely guess which one is the total.
    """
    found = set()
    for match in _STATED_TOTAL_RE.finditer(text):
        raw = match.group(1) or match.group(2) or match.group(3)
        try:
            found.add(float(raw.replace(",", "")))
        except ValueError:
            continue
    if len(found) == 1:
        return found.pop()
    return None


def extract_po_fields(question):
    """
    Pulls structured PO fields (vendor/department/items/justification)
    out of a natural-language PO_REQUEST request.

    Returns:
        (fields, error) - exactly one of these is set. `fields` has
        the keys apply_po() expects (user/vendor/department/items/
        justification), plus the validation results (ok/errors/
        warnings/info) so the confirmation card can show them
        without a second round-trip.
    """

    extraction_prompt = f"""
Extract the fields needed to raise a Purchase Order from the request
below. Reply with ONLY a JSON object, no other text, in exactly this
shape:

{{"vendor": "...", "vendor_email": "...", "department": "...", "items": [{{"name": "...", "quantity": 1, "unit_price": 0.0}}], "justification": "..."}}

- "vendor": the supplier/vendor name.
- "vendor_email": the vendor/seller's email address if the user
  mentioned one, else "".
- "department": the cost-center/department the spend is charged to,
  or "" if not mentioned.
- "items": one entry per distinct item mentioned, with "quantity" as
  a plain number and "unit_price" as a plain number (no currency
  symbols). If only a total was given for one item with no explicit
  unit price, divide the total by the quantity to get the unit
  price.
- "justification": a short free-text business reason if the user
  gave one, else "".

REQUEST:
{question}

JSON:""".strip()

    try:
        raw = _call_extraction_model(extraction_prompt, num_predict=300)
    except Exception as error:
        log_timing(f"extract_po_fields FAILED: {error}")
        return None, f"Couldn't reach the local model to read that request: {error}"

    parsed = _parse_json_object(raw)

    if not parsed:
        return None, "I couldn't parse the details of that PO request."

    vendor = str(parsed.get("vendor", "")).strip()

    # SECURITY/CORRECTNESS: never trust an LLM-generated PO recipient.
    # The recipient must come from an email address literally present in
    # the user's original request. If none was typed, leave it blank and
    # let the confirmation form use the configured fallback.
    literal_emails = re.findall(
        r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+",
        question,
    )
    vendor_email = literal_emails[0].strip() if literal_emails else ""

    department = str(parsed.get("department", "")).strip()
    justification = str(parsed.get("justification", "")).strip()
    raw_items = parsed.get("items", [])

    if not isinstance(raw_items, list):
        raw_items = []

    items = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        items.append({
            "name": str(raw_item.get("name", "")).strip(),
            "quantity": raw_item.get("quantity", 0),
            "unit_price": raw_item.get("unit_price", 0),
        })

    if not vendor:
        return None, "I couldn't tell which vendor that PO is for - could you name the vendor?"

    if not items:
        return None, "I couldn't tell what's being ordered - could you list the item(s), quantity, and unit price?"

    # Sanity-check the extraction against any total the user actually
    # stated in plain text (e.g. "for 30,000 rupees"). The extraction
    # model is asked to derive unit_price = total / quantity when only
    # a total is given, but it can still get that division wrong -
    # this catches a single-item PO where the resulting line total
    # doesn't match what the user said and corrects it instead of
    # silently sending the wrong amount for approval.
    correction_warning = None
    stated_total = _extract_stated_total(question)
    if stated_total is not None and len(items) == 1:
        try:
            quantity = float(items[0]["quantity"] or 0)
        except (TypeError, ValueError):
            quantity = 0
        try:
            computed_total = quantity * float(items[0]["unit_price"] or 0)
        except (TypeError, ValueError):
            computed_total = 0
        if quantity > 0 and abs(computed_total - stated_total) > 0.01:
            corrected_unit_price = round(stated_total / quantity, 2)
            correction_warning = (
                f"Unit price was recalculated to ₹{corrected_unit_price:,.2f} "
                f"to match the ₹{stated_total:,.2f} total you mentioned."
            )
            items[0]["unit_price"] = corrected_unit_price

    # "me" - the primary configured user, same single-tenant
    # convention as extract_leave_fields()/schedule_meeting().
    user = "me"

    ok, errors, warnings, info = validate_po_request(
        user, vendor, department, items, justification
    )
    if correction_warning:
        warnings = [correction_warning] + list(warnings or [])

    return {
        "user": user,
        "vendor": vendor,
        "vendor_email": vendor_email or DEFAULT_VENDOR_EMAIL,
        "department": department or "general",
        "items": info["items"] or items,
        "justification": justification,
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }, None


def _format_action_draft(action_route, fields):
    """
    Formats the confirmation-card text shown before a SEND_MAIL or
    SCHEDULE_MEETING action is actually carried out.
    """

    if action_route == "SEND_MAIL":

        cc_line = f"  \n**Cc:** {fields['cc']}" if fields.get("cc") else ""
        bcc_line = f"  \n**Bcc:** {fields['bcc']}" if fields.get("bcc") else ""

        return (
            "Here's the draft — want me to send it?\n\n"
            f"**To:** {fields['to']}{cc_line}{bcc_line}  \n"
            f"**Subject:** {fields['subject']}\n\n"
            f"{fields['body']}"
        )

    if action_route == "LEAVE_REQUEST":

        start_date = fields["start_date"]
        end_date = fields["end_date"]
        when_line = (
            start_date.strftime("%a, %b %d %Y")
            if start_date == end_date
            else f"{start_date.strftime('%a, %b %d %Y')} – {end_date.strftime('%a, %b %d %Y')}"
        )
        reason_line = f"  \n**Reason:** {fields['reason']}" if fields.get("reason") else ""

        info = fields.get("info") or {}
        balance_line = (
            f"  \n**Balance before:** {info.get('balance_before', '?')} · "
            f"**Days requested:** {info.get('days_requested', '?')}"
        )

        error_block = (
            "\n\n🚫 **This request can't be submitted as-is:**\n"
            + "\n".join(f"- {e}" for e in fields.get("errors") or [])
            if fields.get("errors")
            else ""
        )
        warning_block = (
            "\n\n⚠️ **Note:**\n"
            + "\n".join(f"- {w}" for w in fields.get("warnings") or [])
            if fields.get("warnings")
            else ""
        )

        intro = (
            "Here's the leave request — want me to send it to your "
            "leave approver?"
            if fields.get("ok")
            else "Here's the leave request, but it has problems that need fixing first:"
        )

        return (
            f"{intro}\n\n"
            f"**Type:** {fields['leave_type'].title()} leave  \n"
            f"**When:** {when_line}"
            f"{reason_line}"
            f"{balance_line}"
            f"{error_block}{warning_block}"
        )

    if action_route == "PO_REQUEST":

        info = fields.get("info") or {}
        items = fields.get("items") or []

        items_lines = "\n".join(
            f"- {item['name']}: {item['quantity']} x "
            f"₹{item['unit_price']:,.2f} = ₹{item['line_total']:,.2f}"
            for item in items
        )

        justification_line = (
            f"  \n**Justification:** {fields['justification']}"
            if fields.get("justification")
            else ""
        )

        budget_line = ""
        if info.get("department_limit") is not None:
            budget_line = (
                f"  \n**{fields['department'].title()} budget:** "
                f"₹{info.get('department_spend_before', 0):,.2f} committed → "
                f"₹{info.get('department_spend_after', 0):,.2f} after this PO "
                f"(limit ₹{info['department_limit']:,.2f})"
            )

        error_block = (
            "\n\n🚫 **This PO can't be submitted as-is:**\n"
            + "\n".join(f"- {e}" for e in fields.get("errors") or [])
            if fields.get("errors")
            else ""
        )
        warning_block = (
            "\n\n⚠️ **Note:**\n"
            + "\n".join(f"- {w}" for w in fields.get("warnings") or [])
            if fields.get("warnings")
            else ""
        )

        intro = (
            "Here's the PO — want me to send it to the vendor?"
            if fields.get("ok")
            else "Here's the PO, but it has problems that need fixing first:"
        )

        return (
            f"{intro}\n\n"
            f"**Vendor:** {fields['vendor']}  \n"
            f"**Vendor email:** {fields.get('vendor_email', DEFAULT_VENDOR_EMAIL)}  \n"
            f"**Department:** {fields['department']}\n\n"
            f"{items_lines}\n\n"
            f"**Total:** ₹{info.get('total_amount', 0):,.2f}"
            f"{justification_line}"
            f"{budget_line}"
            f"{error_block}{warning_block}"
        )

    start = fields["start"]
    end = fields["end"]
    loc_line = f"  \n**Location:** {fields['location']}" if fields.get("location") else ""
    att_line = f"  \n**Attendees:** {fields['attendees']}" if fields.get("attendees") else ""

    # Populated by the caller (see the SCHEDULE_MEETING action-agent
    # block in main()) with the result of check_group_availability()
    # against "me" (the primary user's own calendar - see the call
    # site) plus any named attendees' configured calendars - shown
    # here so the user sees a clash BEFORE confirming, not after.
    conflict_lines = []
    for attendee_name, overlapping_events in (fields.get("conflicts") or {}).items():
        display_name = "You" if attendee_name == "me" else attendee_name
        conflict_lines.append(
            f"- **{display_name}** {'are' if display_name == 'You' else 'is'} busy: "
            f"{'; '.join(overlapping_events)}"
        )
    conflict_block = (
        "\n\n⚠️ **Possible conflict:**\n" + "\n".join(conflict_lines)
        if conflict_lines
        else ""
    )

    past_block = (
        "\n\n⚠️ **That time has already passed** - double check the "
        "date/time before scheduling."
        if fields.get("in_past")
        else ""
    )

    return (
        "Here's what I'll add to the calendar — want me to schedule it?\n\n"
        f"**Title:** {fields['title']}  \n"
        f"**When:** {start.strftime('%a, %b %d %Y')} · "
        f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"
        f"{loc_line}{att_line}"
        f"{conflict_block}{past_block}"
    )


def _clear_action_edit_state():
    """
    Removes the edit_action_* widget keys used by the confirmation
    form. Streamlit ignores a text_input's `value=` argument once
    its `key` already exists in session_state, so without this the
    next pending action would silently start pre-filled with the
    previous one's edited text instead of its own extracted fields.
    """

    for key in (
        "edit_action_to", "edit_action_cc", "edit_action_bcc",
        "edit_action_subject",
        "edit_action_body", "edit_action_title", "edit_action_date",
        "edit_action_time", "edit_action_duration",
        "edit_action_location", "edit_action_attendees",
        "edit_action_leave_type", "edit_action_leave_start",
        "edit_action_leave_end", "edit_action_leave_reason",
        "edit_action_po_vendor", "edit_action_po_vendor_email",
        "edit_action_po_department",
        "edit_action_po_items_df", "edit_action_po_justification",
        "po_items_editor",
    ):
        st.session_state.pop(key, None)


def _confirm_pending_action():
    """
    Button callback: actually carries out the pending SEND_MAIL or
    SCHEDULE_MEETING action using whatever is currently in the
    editable form fields (which start pre-filled from the extracted
    fields but may have been corrected by the user), then records
    the result as a normal assistant message.
    """

    action = st.session_state.get("pending_action")
    st.session_state.pending_action = None

    if not action:
        _clear_action_edit_state()
        return

    try:
        if action["kind"] == "SEND_MAIL":

            to = st.session_state.get("edit_action_to", "").strip()
            body = st.session_state.get("edit_action_body", "").strip()

            if not to:
                success, message = False, "No recipient address was given."
            elif not body:
                success, message = False, "The email body was empty."
            else:
                success, message = send_mail(
                    to=to,
                    subject=st.session_state.get("edit_action_subject", "").strip() or "(no subject)",
                    body=body,
                    cc=st.session_state.get("edit_action_cc", "").strip() or None,
                    bcc=st.session_state.get("edit_action_bcc", "").strip() or None,
                )

        elif action["kind"] == "LEAVE_REQUEST":

            leave_type = st.session_state.get("edit_action_leave_type", "annual").strip().lower()
            start_str = st.session_state.get("edit_action_leave_start", "").strip()
            end_str = st.session_state.get("edit_action_leave_end", "").strip()
            reason = st.session_state.get("edit_action_leave_reason", "").strip()

            try:
                start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
                end_date = datetime.strptime(end_str, "%Y-%m-%d").date()

                success, message, _details = apply_leave(
                    user="me",
                    leave_type=leave_type,
                    start_date=start_date,
                    end_date=end_date,
                    reason=reason,
                    # Re-validated fresh from the edited fields above
                    # (not the original extraction) - force=False so
                    # a genuinely blocking problem (e.g. edited into
                    # a past date, or now over balance) still stops
                    # submission rather than being silently pushed
                    # through just because a draft was shown earlier.
                    force=False,
                )
            except ValueError:
                success, message = False, (
                    "Couldn't read that date - use YYYY-MM-DD."
                )

        elif action["kind"] == "PO_REQUEST":

            vendor = st.session_state.get("edit_action_po_vendor", "").strip()
            vendor_email = (
                st.session_state.get("edit_action_po_vendor_email", "").strip()
                or DEFAULT_VENDOR_EMAIL
            )
            department = st.session_state.get("edit_action_po_department", "").strip()
            justification = st.session_state.get("edit_action_po_justification", "").strip()
            items_df = st.session_state.get("edit_action_po_items_df")

            items = []
            parse_error = None

            if items_df is None or items_df.empty:
                parse_error = "Add at least one item (product name, quantity, unit price)."
            else:
                for row_number, row in enumerate(items_df.to_dict("records"), start=1):
                    name = str(row.get("Product Name", "")).strip()
                    if not name:
                        continue  # skip a fully blank trailing row from the editor
                    try:
                        quantity = float(row.get("Quantity", 0) or 0)
                        unit_price = float(row.get("Unit Price (₹)", 0) or 0)
                    except (TypeError, ValueError):
                        parse_error = f"Row {row_number} ({name}): quantity and unit price must be numbers."
                        break
                    items.append({
                        "name": name,
                        "quantity": quantity,
                        "unit_price": unit_price,
                    })

                if not parse_error and not items:
                    parse_error = "Add at least one item (product name, quantity, unit price)."

            if parse_error:
                success, message = False, parse_error
            else:
                success, message, _details = apply_po(
                    user="me",
                    vendor=vendor,
                    vendor_email=vendor_email,
                    department=department,
                    items=items,
                    justification=justification,
                    # Re-validated fresh from the edited fields above,
                    # same reasoning as LEAVE_REQUEST's force=False.
                    force=False,
                )

        else:

            date_str = st.session_state.get("edit_action_date", "").strip()
            time_str = st.session_state.get("edit_action_time", "").strip()

            try:
                start = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                duration = int(st.session_state.get("edit_action_duration", 30) or 30)
                end = start + timedelta(minutes=duration)

                success, message = schedule_meeting(
                    title=st.session_state.get("edit_action_title", "").strip() or "Meeting",
                    start=start,
                    end=end,
                    location=st.session_state.get("edit_action_location", "").strip(),
                    description="",
                    attendees=st.session_state.get("edit_action_attendees", "").strip() or None,
                )
            except ValueError:
                success, message = False, (
                    "Couldn't read that date/time - use YYYY-MM-DD "
                    "and HH:MM."
                )

    except Exception as error:
        success, message = False, f"Something went wrong: {error}"

    result_text = f"✅ {message}" if success else f"⚠️ {message}"

    st.session_state.messages.append(
        {"role": "model", "content": result_text}
    )
    save_message(st.session_state.current_chat_id, "model", result_text)

    _clear_action_edit_state()


def _cancel_pending_action():
    """
    Button callback: discards the pending action without sending
    anything or touching the calendar file.
    """

    st.session_state.pending_action = None
    _clear_action_edit_state()

    cancel_text = "Okay, cancelled — nothing was sent."

    st.session_state.messages.append(
        {"role": "model", "content": cancel_text}
    )
    save_message(st.session_state.current_chat_id, "model", cancel_text)


def _build_today_status_directive(sources):
    """
    Computes a deterministic, unambiguous free/busy fact for today
    from the already-retrieved event `sources` (each a "TITLE
    (YYYY-MM-DD HH:MM)[...]" label) and returns it as a directive
    line to prepend to the calendar evidence.

    Without this, "am I free today?" was left entirely to the LLM
    to infer from a list of raw events - and on questions like this
    the model sometimes answered "you are free" even when a same-day
    event was right there in the evidence (and sometimes dropped a
    later-in-the-day event when listing what's on the calendar).
    Computing the actual answer in Python and stating it as a fact
    removes that guesswork - the model only has to relay it.

    Returns "" if there are no sources to check (nothing to add).
    """

    if not sources:
        return ""

    today_str = datetime.now().strftime("%Y-%m-%d")

    todays_events = []

    for source in sources:
        match = re.search(r"\((\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})\)", source)

        if not match:
            continue

        event_date, event_time = match.groups()

        if event_date == today_str:
            title = source.split(" (")[0]
            todays_events.append((event_time, title))

    if not todays_events:
        return f"TODAY'S STATUS: no events found on today's date ({today_str}) in the evidence above - the user is free today.\n\n"

    todays_events.sort()
    listing = "; ".join(f"{title} at {t}" for t, title in todays_events)

    return (
        f"TODAY'S STATUS: {len(todays_events)} event(s) fall on today's "
        f"date ({today_str}): {listing}. The user is NOT fully free "
        "today - state this plainly and list every one of these "
        "events, noting which (if any) have already passed relative "
        "to the current time.\n\n"
    )


def _build_mail_found_directive(sources):
    """
    Prepends a blunt, unambiguous fact about how many emails were
    retrieved, so the model can't deny finding anything it was
    actually handed. Mirrors _build_today_status_directive() below
    for the same reason: on "summarize the last email" / "did I get
    anything from X" style questions, a small model sometimes claims
    no emails were found even when real EMAIL EVIDENCE is sitting
    right there in the prompt - stating the count as a hard fact
    closes that gap.

    Returns "" if there are no sources (nothing to reinforce).
    """

    if not sources:
        return ""

    listing = "; ".join(sources)

    return (
        f"MAIL SEARCH RESULT: {len(sources)} matching email(s) were "
        f"found and are detailed below: {listing}. Do NOT say no "
        "emails were found - summarize from the evidence below.\n\n"
    )


def build_routed_prompt(question, conversation_history):
    """
    Shared routing + evidence-retrieval + prompt-construction logic.

    Classifies the question (DOCUMENT / MAIL / MEETINGS / WEB / CHAT),
    pulls the right evidence (documents, inbox, calendar, or web),
    and builds the final evidence-grounded prompt text - the same
    prompt regardless of which LLM ends up answering it. Originally
    this lived only inside stream_ollama_response(), which meant
    Gemini/Groq answers never got mail/calendar/web-routed context,
    only document RAG. Both stream_ollama_response() and
    groq_stream_response() call this now so routing works the same
    no matter which model is answering.

    Also sets st.session_state.last_route / last_sources so the UI's
    route badge and Sources expander work for any caller.

    Returns:
        route: one of DOCUMENT / MAIL / MEETINGS / WEB / CHAT
        sources: list of source labels for the UI
        prompt: the fully-built, evidence-grounded prompt text
    """

    # =========================================================
    # SPECULATIVE RETRIEVAL
    #
    # Start BOTH retrieval systems immediately.
    #
    # While the router is deciding:
    #
    #     DuckDuckGo -> searching
    #     RAG        -> retrieving
    #     Router     -> classifying
    #
    # Whichever agent is selected already has its result
    # available or partially completed.
    # =========================================================

    retrieval_executor = ThreadPoolExecutor(
        max_workers=2
    )

    web_search_future = retrieval_executor.submit(
        web_search,
        question,
        4,
    )

    document_future = retrieval_executor.submit(
        retrieve_context,
        question,
        number_of_results=2,
    )

    # =========================================================
    # QUERY ROUTING AGENT
    # =========================================================

    route = route_query(question, conversation_history)

    context = ""
    sources = []

    retrieval_start = time.perf_counter()

    # =========================================================
    # DOCUMENT AGENT
    # =========================================================

    if route == "DOCUMENT":

        try:
            context, sources = document_future.result(
                timeout=20
            )

        except Exception as error:

            log_timing(
                f"retrieve_context FAILED: {error}"
            )

            context = ""
            sources = []

        log_timing(
            f"document retrieval resolved in "
            f"{time.perf_counter() - retrieval_start:.2f}s"
        )

    # =========================================================
    # WEB AGENT
    # =========================================================

    elif route == "WEB":

        try:
            context, sources = web_search_future.result(
                timeout=20
            )

        except Exception as error:

            log_timing(
                f"web_search FAILED: {error}"
            )

            context = ""
            sources = []

        log_timing(
            f"web search resolved in "
            f"{time.perf_counter() - retrieval_start:.2f}s"
        )

    # =========================================================
    # MAIL AGENT
    #
    # Not speculative like web/document above - IMAP is a real
    # network round trip to an external mailbox, and most turns
    # aren't mail questions. Only pay that cost when the route
    # actually resolves to MAIL.
    # =========================================================

    elif route == "MAIL":

        try:
            context, sources = search_mail(question)

            if context:
                context = _build_mail_found_directive(sources) + context

        except Exception as error:

            log_timing(f"search_mail FAILED: {error}")

            context = ""
            sources = []

        log_timing(
            f"mail search resolved in "
            f"{time.perf_counter() - retrieval_start:.2f}s"
        )

    # =========================================================
    # MEETINGS AGENT
    #
    # Also not speculative, for the same reason - no point parsing
    # the calendar file on every turn that isn't calendar-related.
    # =========================================================

    elif route == "MEETINGS":

        try:
            # If the question names one or more people who have a
            # configured calendar (NOVA_MEETINGS_ICS_PATHS), check
            # THEIR calendars specifically - e.g. "is Priya free
            # Thursday?" or "check Priya and Dev's calendars" -
            # instead of only ever looking at the primary user's
            # own calendar.
            mentioned_users = [
                configured_user
                for configured_user in get_configured_calendar_users()
                if configured_user != "me"
                and configured_user in question.lower()
            ]

            if mentioned_users:
                context, sources = search_meetings_multi(
                    question, mentioned_users
                )
            else:
                context, sources = search_meetings(question)

            if context:
                context = _build_today_status_directive(sources) + context

        except Exception as error:

            log_timing(f"search_meetings FAILED: {error}")

            context = ""
            sources = []

        log_timing(
            f"meetings search resolved in "
            f"{time.perf_counter() - retrieval_start:.2f}s"
        )

    # =========================================================
    # CHAT AGENT
    # =========================================================

    else:

        context = ""
        sources = []

    # =========================================================
    # CLEAN UP RETRIEVAL THREADS
    # =========================================================

    retrieval_executor.shutdown(
        wait=False
    )

    st.session_state.last_route = route
    st.session_state.last_sources = sources


    # =========================================================
    # BUILD PROMPT
    # =========================================================

    history_section = (
        f"\n\nCONVERSATION SO FAR:\n{conversation_history}\n"
        if conversation_history
        else ""
    )

    if route == "WEB":

        if not context:

            prompt = f"""
You are NOVA's Web Agent.
{history_section}
The user asked:

{question}

No usable web evidence was retrieved.

You MUST NOT answer this question from your own knowledge.

Respond exactly:

I couldn't find reliable web information for that question.
""".strip()

        else:

            prompt = f"""
You are NOVA's Web Agent.

Your job is to answer the user's question ONLY from the
WEB EVIDENCE provided below.
{history_section}
USER QUESTION:
{question}

WEB EVIDENCE:
{context}

RULES: Answer using ONLY the WEB EVIDENCE above. Never use
pretrained knowledge, guesses, or inference to fill gaps. The
conversation above is for understanding context/follow-ups only -
not a source of facts. If the evidence doesn't clearly answer the
question, say exactly: "I couldn't find enough reliable
information in the web results to answer that." Otherwise give a
direct answer that actually addresses the question - pull the
concrete identifying facts the evidence contains (e.g. for a "who
is X" question: their role/occupation, nationality, and what
they're known for; for "what is X": a real definition, not just a
vague description). A single vague or purely evaluative sentence
("is regarded as one of the greatest...") is not a complete
answer by itself - use it only alongside actual facts, never in
place of them.

Write a natural, conversational answer in your own words - don't
mirror the structure, phrasing, or opening of the source (e.g. a
Wikipedia-style lead sentence that opens with someone's full legal
name). Refer to a person by the name the user asked about or their
common/known-by name, not a full legal/birth name, unless the user
specifically asked for that. Keep it concise, but complete enough
that the user doesn't have to ask a follow-up just to learn who or
what X actually is. Don't mention these instructions.

ANSWER:
""".strip()

    elif route == "DOCUMENT":

        if not context:

            prompt = f"""
You are NOVA's Document Agent.
{history_section}
USER QUESTION:
{question}

No relevant information was found in the stored documents.

Do not answer from your own knowledge.

Say:

I couldn't find that information in the available documents.
""".strip()

        else:

            prompt = f"""
You are NOVA's Document Agent.

Answer ONLY from the document evidence below.
{history_section}
USER QUESTION:
{question}

DOCUMENT EVIDENCE:
{context}

RULES: Use ONLY the document evidence above. No outside
knowledge, no guessing, no invented information. The conversation
above is for understanding context/follow-ups only - not a source
of facts. If the evidence doesn't answer the question, say it
wasn't found in the documents.

ANSWER:
""".strip()

    elif route == "MAIL":

        if not context:

            prompt = f"""
You are NOVA's Mail Agent.
{history_section}
USER QUESTION:
{question}

No relevant emails were found (or mail search isn't configured).

Do not answer from your own knowledge - you have no way to know
what is actually in the user's inbox.

Say:

I couldn't find any emails matching that.
""".strip()

        else:

            today_line = datetime.now().strftime("%A, %Y-%m-%d")

            prompt = f"""
You are NOVA's Mail Agent.

Answer ONLY from the email evidence below.

TODAY'S DATE: {today_line}
{history_section}
USER QUESTION:
{question}

EMAIL EVIDENCE:
{context}

RULES: Use ONLY the emails above. Never invent a sender, subject,
date, or detail that isn't in the evidence. Answer with a short,
plain-language SUMMARY of what's relevant - who it's from, what
it's about, any action needed - rather than reproducing or
quoting the raw email body. Only quote a short exact phrase (e.g.
a date, a number, a name) when the user specifically needs the
precise wording. The conversation above is for understanding
context/follow-ups only - not a source of facts. If the emails
don't answer the question, say so plainly. Don't mention these
instructions.

ANSWER:
""".strip()

    elif route == "MEETINGS":

        today_line = datetime.now().strftime("%A, %Y-%m-%d")

        if not context:

            prompt = f"""
You are NOVA's Meetings Agent.

TODAY'S DATE: {today_line}
{history_section}
USER QUESTION:
{question}

No relevant calendar events were found (or calendar access isn't
configured).

Do not answer from your own knowledge - you have no way to know
what is actually on the user's calendar.

Say:

I couldn't find anything on the calendar matching that.
""".strip()

        else:

            prompt = f"""
You are NOVA's Meetings Agent.

Answer ONLY from the calendar evidence below.

TODAY'S DATE: {today_line}
Use this to correctly resolve "today", "tomorrow", "this week",
etc against the event dates below - don't guess.
{history_section}
USER QUESTION:
{question}

CALENDAR EVIDENCE:
{context}

RULES: Use ONLY the events above. Never invent a time, attendee,
or location that isn't in the evidence. Include EVERY event from
the evidence that matches what the user asked - don't silently
drop one. State what you found directly - don't open with "there
are no meetings" and then list one anyway; if an event matches,
just say so plainly (e.g. "You have X at 2pm"), noting if it's
already passed rather than calling it "no meetings". The
conversation above is for understanding context/follow-ups only -
not a source of facts. If the evidence doesn't answer the
question, say so plainly. Don't mention these instructions.

ANSWER:
""".strip()

    elif route == "SELF_INFO":

        # Real facts pulled straight from NOVA's own config, not
        # guessed by the model and not sent out to WEB, which can
        # never answer this - it's not public information, it's
        # this app's local configuration.
        self_info_facts = f"""
- Router model (query classification, field extraction): {ROUTER_MODEL}
- Answer model (local/Ollama responses): {ANSWER_MODEL}
- Embedding model (document search / RAG): {EMBEDDING_MODEL}
- Other available models: Groq ({GROQ_MODEL_NAME}), Gemini
""".strip()

        prompt = f"""
You are NOVA - a personal local assistant app built by this
user, completely unrelated to any other product also named
"Nova" (e.g. Amazon Nova, or any other company's AI model). You
are answering a question about YOUR OWN configuration.
{history_section}
USER QUESTION:
{question}

REAL FACTS ABOUT NOVA'S SETUP:
{self_info_facts}

RULES: Answer using ONLY the facts above. Don't guess or add
anything not listed. NEVER attribute NOVA to Amazon, or any
other company - NOVA is this user's own app, not a commercial
product, and has no brand/company affiliation. Don't mention
Amazon, AWS, or any other AI vendor unless the facts above
actually name one (e.g. Groq/Gemini as available models is
fine - inventing a company that made "NOVA" itself is not).
Keep it short and direct.

ANSWER:
""".strip()

    else:

        prompt = f"""
You are NOVA, a helpful AI assistant.
{history_section}
USER:
{question}

Answer naturally and concisely.
""".strip()

    st.session_state.last_route = route
    st.session_state.last_sources = sources

    return route, sources, prompt


def stream_ollama_response(question):
    import json

    request_start = time.perf_counter()

    conversation_history = build_local_history()

    route, sources, prompt = build_routed_prompt(
        question, conversation_history
    )

    # =========================================================
    # GENERATION
    #
    # This was previously missing: the grounded prompt above was
    # built but never sent to the model, so NOVA was generating
    # from an empty/undefined stream instead of the routed,
    # evidence-constrained prompt. That's the main hallucination
    # source for the local model.
    # =========================================================

    generation_start = time.perf_counter()
    first_token = True

    # Deterministic for grounded fact-extraction so the model
    # doesn't drift on numbers/specifics; only plain chat gets
    # any warmth.
    generation_temperature = 0.0 if route in ("WEB", "DOCUMENT", "MAIL", "MEETINGS", "SELF_INFO") else 0.5

    response = OLLAMA_SESSION.post(
        OLLAMA_URL,
        json={
            "model": ANSWER_MODEL,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "24h",
            "options": {
                "temperature": generation_temperature,
                "num_ctx": LOCAL_MODEL_NUM_CTX,
                "num_predict": 400,
                "num_gpu": 99,
            },
        },
        stream=True,
        timeout=120,
    )

    response.raise_for_status()

    for line in response.iter_lines():

        if not line:
            continue

        data = json.loads(line)
        token = data.get("response", "")

        if token:
            if first_token:
                first_token = False
                log_timing(
                    f"[ollama] time to first token = "
                    f"{time.perf_counter() - generation_start:.2f}s"
                )
            yield token

        if data.get("done"):
            break

    log_timing(
        f"[ollama] full generation took "
        f"{time.perf_counter() - generation_start:.2f}s | "
        f"TOTAL request = {time.perf_counter() - request_start:.2f}s"
    )
# ---------------------------------------------------

def generate_followup_questions(question, answer, selected_model, api_key):

    trimmed_answer = answer[:400]

    prompt = f"""Question: {question}
Answer: {trimmed_answer}

Write exactly 3 short natural follow-up questions the user might
ask NEXT, based specifically on the content of the answer above -
not generic questions about the conversation itself (e.g. never
write things like "what would you like to discuss next?" or "is
there anything else you're curious about?").
One per line. No numbers, no bullets, no extra text."""

    if selected_model == LOCAL_MODEL_NAME:
        response = OLLAMA_SESSION.post(
            OLLAMA_URL,
            json={
                "model": ROUTER_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "24h",
                "options": {
                    "num_ctx": LOCAL_MODEL_NUM_CTX,
                    "num_predict": 40,
                    "num_gpu": 99,
                },
            },
        )

        response.raise_for_status()
        text = response.json()["response"]

    elif selected_model == GROQ_MODEL_NAME:
        response = GROQ_SESSION.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": selected_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "max_tokens": 60,
            },
            timeout=30,
        )

        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]

    else:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=selected_model,
            contents=prompt,
        )

        text = response.text or ""

    # A small/chatty model sometimes ignores "no extra text" and
    # prepends a preamble line ("Sure, here are three follow-up
    # questions:") before the actual questions. Without filtering,
    # that preamble gets treated as if it were a 4th question and
    # shown as its own clickable button (as happened here). Strip
    # leading numbering/bullets, then drop any line that looks like
    # meta commentary rather than an actual question.
    preamble_re = re.compile(
        r"^(sure|okay|ok|certainly|of course|here are|here's|"
        r"these are|three (short )?(natural )?follow-?up questions|"
        r"follow-?up questions)\b",
        re.IGNORECASE,
    )

    # Generic meta-questions about the CONVERSATION ITSELF ("what
    # would you like to discuss next?", "anything else you're
    # curious about?") rather than about the actual content of the
    # answer - a small model falls back to these as filler when it
    # doesn't have (or doesn't use) real substance to build a
    # follow-up from. They're technically well-formed questions, so
    # the preamble filter above doesn't catch them - filter on
    # meaning instead.
    generic_filler_re = re.compile(
        r"(what (would|do) you (like|want) to (discuss|talk about|"
        r"know|do) next|"
        r"(are you )?looking for (information|more info) on a "
        r"specific topic|"
        r"is there anything (else )?you'?re curious about|"
        r"anything else (i can help|you'?d like to know|you need "
        r"help with)|"
        r"do you have any other questions|"
        r"what (else )?can i (help|do) (for|with) you|"
        r"is there (anything|something) (specific )?(you'?d like|"
        r"you want) (to (know|ask))?)",
        re.IGNORECASE,
    )

    questions = []

    for raw_line in text.splitlines():

        line = raw_line.strip()
        line = re.sub(r"^[\-\*\u2022]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = line.strip()

        if not line:
            continue

        if preamble_re.match(line):
            continue

        # A preamble sentence ending in ":" with no "?" ("Here's
        # what you could ask:") - a real follow-up question doesn't
        # look like this.
        if line.endswith(":") and "?" not in line:
            continue

        if generic_filler_re.search(line):
            continue

        questions.append(line)

    return questions[:3]


# ---------------------------------------------------
# Background follow-up generation
#
# generate_followup_questions() only ever looks at the first
# 400 characters of the answer, so it doesn't need to wait for
# the main answer to finish streaming - just for enough of it
# to exist. teed_stream() watches the tokens as they pass
# through to st.write_stream and flips an Event once the
# threshold is hit (or the stream ends, for short answers).
# safe_generate_followups() waits on that Event and then makes
# the follow-up call on a background thread, in parallel with
# the rest of the main answer streaming in.
#
# Note: safe_generate_followups() must never call st.* - only
# the main script thread has Streamlit's render context.
# ---------------------------------------------------

def teed_stream(source_generator, buffer_holder, ready_event, threshold=400):
    """Pass chunks through unchanged, signalling once enough text has accumulated."""

    for chunk in source_generator:

        buffer_holder["text"] += chunk

        if not ready_event.is_set() and len(buffer_holder["text"]) >= threshold:
            ready_event.set()

        yield chunk

    # Stream ended before hitting the threshold (short answer) -
    # release the waiting thread with whatever text we have.
    ready_event.set()


def safe_generate_followups(
    question,
    buffer_holder,
    ready_event,
    selected_model,
    api_key,
    result_holder,
):
    """Runs on a background thread - must not touch st.* directly."""

    ready_event.wait(timeout=30)

    try:
        result_holder["questions"] = generate_followup_questions(
            question,
            buffer_holder["text"],
            selected_model,
            api_key,
        )
    except Exception:
        result_holder["questions"] = []


# Interface
# ---------------------------------------------------
def render_sidebar(api_key):
    with st.sidebar:

        st.markdown(
            """
            <div class="nova-sidebar-logo">
                <span class="nova-sidebar-star">✦</span>
                NOVA
            </div>

            <div class="nova-sidebar-subtitle">
                Your private AI workspace
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-section-title">AI Mode</div>',
            unsafe_allow_html=True,
        )

        mode = st.selectbox(
            "AI Mode",
            ["Ollama (Local)", "Gemini", "Groq"],
            label_visibility="collapsed",
        )

        if mode == "Gemini":
            model_name = "gemini-3.6-flash"
        elif mode == "Groq":
            model_name = GROQ_MODEL_NAME
        else:
            model_name = LOCAL_MODEL_NAME

        # ================================
        # LEAVE REQUEST STATUS
        #
        # Read-only status list in the sidebar - always visible (NOT
        # gated behind Admin Access, since the leave approver isn't
        # necessarily the same person as the knowledge-base admin).
        # There's no approve/reject action here: approval is handled
        # elsewhere (e.g. by whoever owns approve_leave_request()/
        # reject_leave_request() - a future admin flow, a script, a
        # different surface). This list just reflects whatever that
        # status currently is - a request shows "Pending" until it's
        # acted on, then flips to "Approved"/"Rejected" and stays
        # visible instead of disappearing off the list.
        # ================================

        # ================================
        # LEAVE REQUEST STATUS
        #
        # Collapsed into a single expander (one "dropdown" for the
        # whole list, not one per request) so the sidebar doesn't
        # grow a row per request by default - opening it reveals
        # every request at once. Always visible (NOT gated behind
        # Admin Access, since the leave approver isn't necessarily
        # the same person as the knowledge-base admin). There's no
        # approve/reject action here: approval is handled elsewhere
        # (e.g. by whoever owns approve_leave_request()/
        # reject_leave_request() - a future admin flow, a script, a
        # different surface). This list just reflects whatever that
        # status currently is - a request shows "Pending" until it's
        # acted on, then flips to "Approved"/"Rejected" and stays
        # visible instead of disappearing off the list.
        # ================================

        # Most recent first, capped so a busy history doesn't take
        # over the whole sidebar.
        all_leave_requests = get_all_leave_requests()[:15]

        LEAVE_STATUS_STYLE = {
            "pending": ("Pending", "#c98a2b"),
            "approved": ("Approved", "#3fae5c"),
            "rejected": ("Rejected", "#d1495b"),
        }

        with st.expander(f"Leave Requests ({len(all_leave_requests)})"):

            if not all_leave_requests:
                st.markdown(
                    '<div style="color:#eeeeef; font-size:0.85rem;">'
                    "No leave requests yet.</div>",
                    unsafe_allow_html=True,
                )
            else:
                for leave_request in all_leave_requests:

                    status_label, status_color = LEAVE_STATUS_STYLE.get(
                        leave_request.get("status"),
                        (leave_request.get("status", "").title(), "#8a8a94"),
                    )

                    st.markdown(
                        f"""
                        <div style="color:#eeeeef; font-size:0.85rem; margin-bottom:0.6rem;">
                            <strong>{leave_request['user']}</strong> ·
                            {leave_request['leave_type'].title()} ·
                            {leave_request['start']} to {leave_request['end']}
                            ({leave_request['days']}d)
                            <span style="
                                display:inline-block;
                                margin-left:0.35rem;
                                padding:0.05rem 0.5rem;
                                border-radius:1rem;
                                font-size:0.72rem;
                                font-weight:600;
                                color:#ffffff;
                                background:{status_color};
                            ">{status_label}</span>
                            {f"<br/><span style='color:#b8b8c2;'>{leave_request['reason']}</span>" if leave_request.get('reason') else ""}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if st.button("🗑 Clear Leave Requests", use_container_width=True):
            if clear_leave_requests():
                st.success("Leave requests cleared.")
            else:
                st.error("Couldn't clear leave requests.")
            st.rerun()

        # ================================
        # PURCHASE ORDER STATUS
        #
        # Same read-only pattern as the Leave Requests list above:
        # no approve/reject action here, just a status list that
        # flips Pending -> Approved/Rejected (or shows
        # "Auto-approved" for POs that cleared under
        # NOVA_PO_AUTO_APPROVE_THRESHOLD without needing sign-off).
        # ================================

        # ================================
        # PURCHASE ORDER STATUS
        #
        # Same collapsed-expander pattern as the Leave Requests list
        # above: no approve/reject action here, just a status list
        # that flips Pending -> Approved/Rejected (or shows
        # "Auto-approved" for POs that cleared under
        # NOVA_PO_AUTO_APPROVE_THRESHOLD without needing sign-off).
        # ================================

        # ================================
        # PO APPROVAL IS DONE FROM THE EMAIL. Only sent POs are shown in the sidebar.

        sent_po_requests = get_sent_po_requests()[:15]

        PO_STATUS_STYLE = {
            "pending": ("Pending", "#c98a2b"),
            "approved": ("Approved", "#3fae5c"),
            "auto_approved": ("Auto-approved", "#3fae5c"),
            "rejected": ("Rejected", "#d1495b"),
        }

        with st.expander(f"Sent Purchase Orders ({len(sent_po_requests)})"):
            if not sent_po_requests:
                st.markdown(
                    '<div style="color:#eeeeef; font-size:0.85rem;">'
                    "No sent POs yet.</div>",
                    unsafe_allow_html=True,
                )
            else:
                for po_request in sent_po_requests:
                    item_count = len(po_request.get("items", []))
                    item_summary = (
                        f"{item_count} item{'s' if item_count != 1 else ''}"
                    )
                    st.markdown(
                        f"""
                        <div style="color:#eeeeef; font-size:0.85rem; margin-bottom:0.6rem;">
                            <strong>{po_request['requester']}</strong> ·
                            {po_request['vendor']} ·
                            ₹{po_request.get('total_amount', 0):,.2f}
                            ({item_summary})
                            <span style="
                                display:inline-block;
                                margin-left:0.35rem;
                                padding:0.05rem 0.5rem;
                                border-radius:1rem;
                                font-size:0.72rem;
                                font-weight:600;
                                color:#ffffff;
                                background:#3fae5c;
                            ">Sent</span>
                            <br/><span style="color:#b8b8c2;">{po_request.get('department', '')}</span>
                            <br/><span style="color:#b8b8c2;">{po_request.get('vendor_email', '')}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if st.button("🗑 Clear POs", use_container_width=True):
            if clear_po_requests():
                st.success("PO history cleared.")
            else:
                st.error("Couldn't clear PO history.")
            st.rerun()

        st.markdown(
            '<div class="sidebar-section-title">Admin Access</div>',
            unsafe_allow_html=True,
        )

        admin_password = st.text_input(
            "Admin password",
            type="password",
            label_visibility="collapsed",
            placeholder="Admin password",
        )

        is_admin = False

        if admin_password:
            if admin_password == os.getenv("ADMIN_PASSWORD"):
                is_admin = True
            else:
                st.error("Incorrect password.")

        if is_admin:

            st.markdown(
                '<div class="sidebar-section-title">Admin Knowledge Base</div>',
                unsafe_allow_html=True,
            )

            admin_file = st.file_uploader(
                "Add permanent document",
                type=["pdf", "txt", "docx", "csv", "xlsx", "doc"],
                key="admin_document_uploader",
            )

            if admin_file is not None:

                if st.button(
                    "Add to knowledge base",
                    use_container_width=True,
                ):
                    try:
                        with st.spinner("Indexing document..."):
                            chunk_count, message = index_admin_document(
                                admin_file
                            )

                        if chunk_count > 0:
                            st.success(
                                f"Added {chunk_count} sections."
                            )
                        else:
                            st.info(message)

                    except Exception as error:
                        st.error(
                            f"Could not add document: {error}"
                        )

            if st.button(
                "Clear documents",
                use_container_width=True,
            ):
                clear_documents()
                st.success("User documents cleared.")

            # ================================
            # INDEXED DOCUMENTS LIST
            #
            # Shows what's actually in the knowledge base right
            # now, not just a confirmation that the last upload
            # worked - useful for checking what NOVA can actually
            # answer DOCUMENT questions from before recording a
            # demo, without guessing from memory.
            # ================================

            indexed_documents = list_indexed_documents()

            with st.expander(
                f"📚 Indexed Documents ({len(indexed_documents)})",
                expanded=False,
            ):
                if not indexed_documents:
                    st.markdown(
                        '<div style="color:#eeeeef; font-size:0.85rem;">'
                        "No documents indexed yet.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    for doc in indexed_documents:
                        label = "Admin" if doc["collection"] == "admin" else "User"
                        st.markdown(
                            f"""
                            <div style="color:#eeeeef; font-size:0.85rem; margin-bottom:0.4rem;">
                                <strong>{doc['source']}</strong><br/>
                                <span style="color:#b8b8c2;">
                                    {doc['chunk_count']} chunk(s) · {label}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        st.markdown(
            '<div class="sidebar-section-title">Chat</div>',
            unsafe_allow_html=True,
        )

        if st.button(
            "＋  New Chat",
            use_container_width=True,
        ):
            clear_chat()
            st.rerun()

        # ================================
        # SAVED CONVERSATIONS
        # ================================

        st.markdown(
            '<div class="sidebar-section-title">Conversations</div>',
            unsafe_allow_html=True,
        )

        chats = get_chats()

        with st.container(key="conversation_list"):

            for chat_id, title, created_at in chats:

                is_current = (
                    chat_id == st.session_state.current_chat_id
                )

                if is_current:
                    button_label = f"●  {title}"
                else:
                    button_label = title

                if st.button(
                    button_label,
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                ):

                    if chat_id != st.session_state.current_chat_id:

                        st.session_state.current_chat_id = chat_id

                        st.session_state.messages = load_chat_messages(
                            chat_id
                        )

                        st.rerun()

        if st.button(
            "⌫  Delete chat history",
            use_container_width=True,
        ):
            delete_chat_history()
            st.success("Chat history deleted.")
            st.rerun()

        st.markdown(
            """
            <div style="
                margin-top:1.5rem;
                padding:1rem;
                border:1px solid rgba(109,69,232,0.35);
                border-radius:12px;
                color:#9999a5;
                font-size:0.78rem;
                line-height:1.5;
            ">
                NOVA retrieves information from its
                stored knowledge base.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return model_name


def main():
    apply_custom_styles()
    initialise_session()

    warm_up_ollama()
    warm_up_embedding_model()

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    if "pending_action" not in st.session_state:
        st.session_state.pending_action = None

    api_key = get_api_key()
    groq_api_key = get_groq_api_key()
    selected_model = render_sidebar(api_key)

    # The single active key for whichever provider is selected -
    # this is what gets passed down to the stream/follow-up calls
    # below, same as before Groq existed (api_key was always
    # "the Gemini key" then; now it's "whichever key is active").
    if selected_model == GROQ_MODEL_NAME:
        active_api_key = groq_api_key
    else:
        active_api_key = api_key

    # ================================
    # TOP BAR
    # ================================

    st.markdown(
        """
        <div class="nova-topbar">
            <div class="nova-top-logo">
                <span>✦</span> NOVA
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

        # ================================
    # EMPTY STATE
    # ================================

    if not st.session_state.messages:

        st.markdown(
            """
            <div class="nova-empty-state">
                <h1>Think it through.</h1>
                <p>
                    A calm space to ask, explore, and create.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    # ================================
    # CHAT HISTORY
    # ================================

    for message in st.session_state.messages:

        display_role = (
            "assistant"
            if message["role"] == "model"
            else "user"
        )

        with st.chat_message(display_role):
            st.markdown(message["content"])

            # Redraw the agent badge/Sources for past assistant
            # messages too - these are now persisted per-message
            # (see save_message/load_chat_messages), not just held
            # in the transient last_route/last_sources session
            # state that only reflected the most-recently-generated
            # reply.
            if display_role == "assistant":

                route_badge = {
                    "DOCUMENT": "📄 Document Agent",
                    "WEB": "🌐 Web Agent",
                    "MAIL": "📧 Mail Agent",
                    "MEETINGS": "📅 Meetings Agent",
                    "CHAT": "💬 Chat",
                    "SELF_INFO": "⚙️ About NOVA",
                }.get(message.get("route"), None)

                if route_badge:
                    st.caption(route_badge)

                message_sources = message.get("sources")

                if message_sources:
                    with st.expander("Sources"):
                        for source in message_sources:
                            st.markdown(f"- {source}")

    # ================================
    # PENDING ACTION CONFIRMATION
    #
    # A SEND_MAIL/SCHEDULE_MEETING draft is waiting on an explicit
    # confirmation before anything actually goes out or gets
    # written to the calendar. While one is pending, hold off on
    # new input so it can't race with a fresh message.
    # ================================

    if st.session_state.get("pending_action"):

        action = st.session_state.pending_action
        fields = action["fields"]

        st.markdown("**Review and edit before sending:**")

        if action["kind"] == "SEND_MAIL":

            st.text_input("To", value=fields.get("to", ""), key="edit_action_to")
            st.text_input("Cc (optional)", value=fields.get("cc") or "", key="edit_action_cc")
            st.text_input("Bcc (optional)", value=fields.get("bcc") or "", key="edit_action_bcc")
            st.text_input("Subject", value=fields.get("subject", ""), key="edit_action_subject")
            st.text_area("Body", value=fields.get("body", ""), key="edit_action_body", height=160)

            confirm_label = "✅ Send it"

        elif action["kind"] == "LEAVE_REQUEST":

            st.selectbox(
                "Leave type",
                ("annual", "sick", "casual"),
                index=("annual", "sick", "casual").index(
                    fields.get("leave_type", "annual")
                ),
                key="edit_action_leave_type",
            )

            date_col, date_col2 = st.columns(2)
            with date_col:
                st.text_input(
                    "Start date (YYYY-MM-DD)",
                    value=fields["start_date"].strftime("%Y-%m-%d"),
                    key="edit_action_leave_start",
                )
            with date_col2:
                st.text_input(
                    "End date (YYYY-MM-DD)",
                    value=fields["end_date"].strftime("%Y-%m-%d"),
                    key="edit_action_leave_end",
                )

            st.text_input(
                "Reason (optional)",
                value=fields.get("reason") or "",
                key="edit_action_leave_reason",
            )

            if fields.get("errors"):
                for err in fields["errors"]:
                    st.error(err)
            if fields.get("warnings"):
                for warn in fields["warnings"]:
                    st.warning(warn)

            confirm_label = "✅ Send to approver"

        elif action["kind"] == "PO_REQUEST":

            st.text_input("Vendor", value=fields.get("vendor", ""), key="edit_action_po_vendor")
            st.text_input(
                "Vendor email (PO will be sent here)",
                value=fields.get("vendor_email", "") or DEFAULT_VENDOR_EMAIL,
                key="edit_action_po_vendor_email",
            )
            st.text_input(
                "Department",
                value=fields.get("department", ""),
                key="edit_action_po_department",
            )

            # A real editable table - separate Product Name/Quantity/
            # Unit Price columns, with add/delete rows built in -
            # instead of asking the user to type "name, qty, price"
            # comma-separated lines. The edited table is captured
            # into a plain session_state variable on every render
            # (rather than relying on the data_editor widget's own
            # keyed state, whose shape - diff dict vs. full frame -
            # varies across Streamlit versions), so
            # _confirm_pending_action always reads a clean DataFrame.
            items_default_df = pd.DataFrame(
                [
                    {
                        "Product Name": item["name"],
                        "Quantity": item["quantity"],
                        "Unit Price (₹)": item["unit_price"],
                    }
                    for item in (fields.get("items") or [])
                ]
                or [{"Product Name": "", "Quantity": 1, "Unit Price (₹)": 0.0}]
            )

            edited_items_df = st.data_editor(
                items_default_df,
                key="po_items_editor",
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Product Name": st.column_config.TextColumn(
                        "Product Name", required=True
                    ),
                    "Quantity": st.column_config.NumberColumn(
                        "Quantity", min_value=0.01, step=1.0, format="%.2f"
                    ),
                    "Unit Price (₹)": st.column_config.NumberColumn(
                        "Unit Price (₹)", min_value=0.0, step=0.01, format="%.2f"
                    ),
                },
            )
            st.session_state["edit_action_po_items_df"] = edited_items_df

            st.text_input(
                "Justification (optional)",
                value=fields.get("justification") or "",
                key="edit_action_po_justification",
            )

            if fields.get("errors"):
                for err in fields["errors"]:
                    st.error(err)
            if fields.get("warnings"):
                for warn in fields["warnings"]:
                    st.warning(warn)

            confirm_label = "✅ Submit for approval"

        else:

            st.text_input("Title", value=fields.get("title", ""), key="edit_action_title")

            date_col, time_col = st.columns(2)
            with date_col:
                st.text_input(
                    "Date (YYYY-MM-DD)",
                    value=fields["start"].strftime("%Y-%m-%d"),
                    key="edit_action_date",
                )
            with time_col:
                st.text_input(
                    "Time (HH:MM)",
                    value=fields["start"].strftime("%H:%M"),
                    key="edit_action_time",
                )

            existing_duration = int(
                (fields["end"] - fields["start"]).total_seconds() // 60
            )
            st.number_input(
                "Duration (minutes)",
                value=existing_duration,
                min_value=5,
                step=5,
                key="edit_action_duration",
            )
            st.text_input(
                "Location (optional)",
                value=fields.get("location") or "",
                key="edit_action_location",
            )
            st.text_input(
                "Attendees (optional)",
                value=fields.get("attendees") or "",
                key="edit_action_attendees",
            )

            confirm_label = "✅ Add to calendar"

        col1, col2 = st.columns(2)

        with col1:
            st.button(
                confirm_label,
                key="confirm_pending_action",
                use_container_width=True,
                on_click=_confirm_pending_action,
            )

        with col2:
            st.button(
                "✖️ Cancel",
                key="cancel_pending_action",
                use_container_width=True,
                on_click=_cancel_pending_action,
            )

        st.caption("Edit any field above, then confirm or cancel before sending another message.")

        return

    # ================================
    # CHAT INPUT
    # ================================

    pending_prompt = st.session_state.pop(
        "pending_prompt",
        None,
    )

    chat_prompt = st.chat_input(
        f"Message {APP_NAME}..."
    )

    prompt = pending_prompt or chat_prompt

    if not prompt:
        return

    # ================================
    # API KEY CHECK
    # ================================

    if selected_model == GROQ_MODEL_NAME and not groq_api_key:
        st.error(
            "No Groq API key found. Set GROQ_API_KEY (or "
            "GROK_API_KEY) in your .env file and restart Streamlit."
        )
        st.stop()
    elif (
        selected_model not in (LOCAL_MODEL_NAME, GROQ_MODEL_NAME)
        and not api_key
    ):
        st.error(
            "GEMINI_API_KEY was not found. "
            "Check your .env file and restart Streamlit."
        )
        st.stop()

    # ================================
    # USER MESSAGE
    # ================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    save_message(
        st.session_state.current_chat_id,
        "user",
        prompt,
    )

    # Create the conversation title from the first user message
    if len(st.session_state.messages) == 1:

        title = prompt.strip()

        if len(title) > 40:
            title = title[:40].rstrip() + "..."

        update_chat_title(
            st.session_state.current_chat_id,
            title,
        )

    with st.chat_message("user"):
        st.markdown(prompt)

    # ================================
    # ACTION AGENTS: SEND MAIL / SCHEDULE MEETING
    #
    # Side-effecting, unlike every other route above - not "answer
    # a question from evidence". Routed the same way as MAIL/
    # MEETINGS, but instead of streaming an LLM answer we extract
    # structured fields, show a confirmation card, and wait for an
    # explicit Confirm click (handled by the PENDING ACTION
    # CONFIRMATION block above) before anything is actually sent
    # or written to the calendar.
    # ================================

    action_route = route_query(prompt, build_local_history())

    if action_route in ("SEND_MAIL", "SCHEDULE_MEETING", "LEAVE_REQUEST", "PO_REQUEST"):

        with st.spinner("Reading that back..."):

            if action_route == "SEND_MAIL":
                fields, error = extract_mail_fields(prompt)
            elif action_route == "LEAVE_REQUEST":
                fields, error = extract_leave_fields(prompt)
            elif action_route == "PO_REQUEST":
                fields, error = extract_po_fields(prompt)
            else:
                fields, error = extract_meeting_fields(prompt)

                # Multi-user schedule check: before drafting the
                # confirmation card, see whether the PRIMARY user
                # (schedule_meeting() always writes the new event to
                # their own calendar, regardless of attendees) or
                # any named attendee has a conflicting event on
                # their OWN configured calendar at the proposed
                # time. "me" is always included here even if the
                # user didn't literally say "with me" - otherwise a
                # request like "schedule a meeting with Kishore at
                # 3pm" only checks Kishore's (usually unconfigured)
                # calendar and silently double-books the primary
                # user's own existing 3pm event, which is the
                # calendar actually being written to. Only names
                # that match a configured calendar (see
                # NOVA_MEETINGS_ICS_PATHS) can actually be checked;
                # everyone else is simply skipped, since there's no
                # way to know their schedule.
                if not error:
                    try:
                        attendees_to_check = "me"
                        if fields.get("attendees"):
                            attendees_to_check += f", {fields['attendees']}"

                        conflicts, _unchecked = check_group_availability(
                            attendees_to_check,
                            fields["start"],
                            fields["end"],
                        )
                    except Exception as availability_error:
                        log_timing(
                            f"check_group_availability FAILED: {availability_error}"
                        )
                        conflicts = {}
                    fields["conflicts"] = conflicts

        if error:

            st.session_state.messages.append(
                {"role": "model", "content": error}
            )
            save_message(st.session_state.current_chat_id, "model", error)
            st.rerun()

        draft_text = _format_action_draft(action_route, fields)

        st.session_state.messages.append(
            {"role": "model", "content": draft_text}
        )
        save_message(st.session_state.current_chat_id, "model", draft_text)

        st.session_state.pending_action = {
            "kind": action_route,
            "fields": fields,
        }

        st.rerun()

    # ================================
    # NOVA RESPONSE
    # ================================

    with st.chat_message("assistant"):

        try:

            status = st.empty()

            status.markdown(
                """
                <div>
                    <span class="nova-loader"></span>
                    <span style="color:#6d45e8;">
                        NOVA is thinking...
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ================================
            # KICK OFF FOLLOW-UPS IN THE BACKGROUND
            #
            # This is submitted immediately, but doesn't
            # actually call the follow-up model until
            # teed_stream() signals that ~400 chars of the
            # answer exist (see safe_generate_followups above).
            #
            # We can't check last_route here to decide whether
            # to skip it - for the local model, last_route is
            # only set once stream_ollama_response() starts
            # running below, so it would still hold last turn's
            # value. Instead we always submit, and decide
            # whether to use the result after the stream (once
            # last_route is current) further down.
            #
            # One exception: a bare greeting ("hi", "thanks",
            # "bye"...) has no real content to build a follow-up
            # question on, so a small model just invents generic
            # filler about the conversation itself ("what would you
            # like to discuss next?") - not useful, and not worth
            # the extra model call. Skip submitting entirely for
            # these; anything with real content still goes through.
            # ================================

            is_bare_greeting = (
                prompt.strip().lower().strip(" !.,?~-") in GREETING_SIGNALS
            )

            followup_buffer = {"text": ""}
            followup_ready = threading.Event()
            followup_result = {}
            followup_executor = ThreadPoolExecutor(max_workers=1)

            if not is_bare_greeting:
                followup_future = followup_executor.submit(
                    safe_generate_followups,
                    prompt,
                    followup_buffer,
                    followup_ready,
                    selected_model,
                    active_api_key,
                    followup_result,
                )
            else:
                followup_future = None

            if selected_model == LOCAL_MODEL_NAME:

                base_stream = stream_ollama_response(
                    prompt
                )

            elif selected_model == GROQ_MODEL_NAME:

                base_stream = groq_stream_response(
                    active_api_key,
                    selected_model,
                    prompt,
                )

            else:

                base_stream = stream_response(
                    active_api_key,
                    selected_model,
                    prompt,
                )

            answer = st.write_stream(
                teed_stream(
                    base_stream,
                    followup_buffer,
                    followup_ready,
                )
            )

            status.empty()

            # All three models now run through build_routed_prompt()
            # (see stream_response()/groq_stream_response()'s
            # docstrings), so last_route/last_sources are populated
            # no matter which one answered - the badge and Sources
            # expander should show for Gemini too, not just local/Groq.

            route_badge = {
                "DOCUMENT": "📄 Document Agent",
                "WEB": "🌐 Web Agent",
                "MAIL": "📧 Mail Agent",
                "MEETINGS": "📅 Meetings Agent",
                "CHAT": "💬 Chat",
                "SELF_INFO": "⚙️ About NOVA",
            }.get(
                st.session_state.get("last_route"),
                None,
            )

            if route_badge:
                st.caption(route_badge)

            sources = st.session_state.get("last_sources")

            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")

            # Captured here (immediately after generation) rather
            # than read again down at the save step below, since
            # last_route/last_sources are single global slots that
            # a later action (e.g. a background follow-up call)
            # could in principle overwrite before we get there.
            answer_route = st.session_state.get("last_route")
            answer_sources = st.session_state.get("last_sources")

        except Exception as error:

            status.empty()

            # Make sure the background follow-up thread doesn't
            # get orphaned if the main stream itself failed.
            if "followup_executor" in locals():
                followup_ready.set()
                followup_executor.shutdown(wait=False)

            st.error(
                f"Could not generate response: {error}"
            )

            return

    # ================================
    # SAVE NOVA RESPONSE
    # ================================

    st.session_state.messages.append(
        {
            "role": "model",
            "content": answer,
            "route": answer_route,
            "sources": answer_sources,
        }
    )

    save_message(
        st.session_state.current_chat_id,
        "model",
        answer,
        route=answer_route,
        sources=answer_sources,
    )

    # ================================
    # FOLLOW-UP QUESTIONS
    # ================================

    # Follow-up suggestions are generated for every agent/route
    # (DOCUMENT, WEB, MAIL, MEETINGS, and CHAT) and for every model -
    # local, Groq, and Gemini alike. There's no route-based skip
    # here anymore; safe_generate_followups() was already kicked off
    # in the background above for every turn EXCEPT bare greetings
    # (followup_future is None for those - see the submission site
    # above), so we just collect whatever's there.
    if followup_future is None:
        followups = []
        followup_executor.shutdown(wait=False)
    else:
        try:
            followup_future.result(timeout=10)
            followups = followup_result.get("questions", [])
        except Exception:
            followups = []
        finally:
            followup_executor.shutdown(wait=False)

    # Nothing real to show - either the greeting-skip above, or the
    # model's suggestions came back empty/all-filler and got
    # filtered out entirely by generate_followup_questions(). Fall
    # back to generic task starters instead of leaving the row
    # empty, so there's always something useful to tap.
    if not followups:
        followups = random.sample(
            STARTER_SUGGESTIONS,
            min(3, len(STARTER_SUGGESTIONS)),
        )

    if followups:

        st.markdown(
            '<div class="followup-title">You might also ask</div>',
            unsafe_allow_html=True,
        )

        with st.container(key="followup_row"):

            columns = st.columns(len(followups))

            for index, followup_question in enumerate(followups):

                with columns[index]:

                    st.button(
                        followup_question,
                        key=f"followup_{index}",
                        use_container_width=True,
                        on_click=lambda q=followup_question: (
                            st.session_state.__setitem__(
                                "pending_prompt",
                                q,
                            )
                        ),
                    )


if __name__ == "__main__":
    main()