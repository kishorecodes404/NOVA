"""
Streamlit AI Chatbot using Google Gemini.

Install:
    pip install streamlit google-genai

Set API key in PowerShell:
    $env:GOOGLE_API_KEY="your_gemini_api_key"

Run:
    streamlit run app.py
"""

import os

import streamlit as st
from google import genai
from dotenv import load_dotenv
load_dotenv()
from rag import index_document, retrieve_context, clear_documents
# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
APP_NAME = "NOVA"

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


# ---------------------------------------------------
# Styling
# ---------------------------------------------------
def apply_custom_styles():
    st.markdown(
        """
        <style>
            :root {
                --ink: #1d1d1f;
                --muted: #6e6e73;
                --surface: #ffffff;
                --canvas: #f5f5f7;
                --accent: #0071e3;
                --line: rgba(0, 0, 0, 0.10);
            }

            /* Make the complete Streamlit app light */
            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"],
            [data-testid="stBottom"] {
                background: var(--canvas) !important;
                color: var(--ink) !important;
            }

            /* Top Streamlit bar */
            [data-testid="stHeader"] {
                background: rgba(255, 255, 255, 0.96) !important;
                border-bottom: 1px solid var(--line);
            }

            /* Main page spacing */
            .block-container {
                max-width: 58rem;
                padding-top: 3.5rem;
                padding-bottom: 8rem;
            }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background: #ffffff !important;
                border-right: 1px solid var(--line);
            }

            [data-testid="stSidebar"] > div:first-child {
                padding-top: 1.5rem;
            }

            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] .stMarkdown {
                color: var(--ink) !important;
            }

            /* Large sidebar app logo */
            .sidebar-logo {
                color: var(--accent);
                font-size: 2.15rem;
                font-weight: 800;
                letter-spacing: -0.06em;
                margin-bottom: 0;
            }

            .sidebar-logo span {
                color: var(--accent) !important;
                font-size: 2.35rem;
                margin-right: 0.35rem;
            }

            .sidebar-tagline {
                color: var(--muted) !important;
                font-size: 0.9rem;
                margin-top: 0.2rem;
                margin-bottom: 1.3rem;
            }

            /* Main logo and heading */
            .nova-brand {
                color: var(--accent);
                font-size: 1.25rem;
                font-weight: 800;
                letter-spacing: 0.16em;
                margin-bottom: 0.7rem;
            }

            .nova-brand span {
                font-size: 1.65rem;
                margin-right: 0.3rem;
            }

            .nova-title {
                color: var(--ink);
                font-size: clamp(2.6rem, 6vw, 4.2rem);
                font-weight: 750;
                letter-spacing: -0.065em;
                line-height: 1.04;
                margin: 0;
            }

            .nova-subtitle {
                color: var(--muted);
                font-size: 1.1rem;
                margin: 0.9rem 0 2.2rem;
            }

            /* Chat messages */
            [data-testid="stChatMessage"] {
                background: var(--surface) !important;
                border: 1px solid var(--line);
                border-radius: 18px;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
                margin: 1rem 0;
                padding: 0.45rem 0.6rem;
            }

            [data-testid="stChatMessage"] p,
            [data-testid="stChatMessage"] li,
            [data-testid="stChatMessage"] span {
                color: var(--ink) !important;
            }

            /* Chat input: removes the dark bottom section */
         /* Modern floating chat input */
[data-testid="stChatInput"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 1rem 0 !important;
}

[data-testid="stChatInput"] > div {
    background: #ffffff !important;
    border: 1px solid rgba(0, 0, 0, 0.12) !important;
    border-radius: 28px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.10) !important;
    padding: 0.35rem 0.45rem 0.35rem 0.8rem !important;
    transition: border 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: #0071e3 !important;
    box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.12) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #1d1d1f !important;
    font-size: 1rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #86868b !important;
}

[data-testid="stChatInput"] button {
    background: #0071e3 !important;
    border-radius: 50% !important;
    color: white !important;
    width: 2.45rem !important;
    height: 2.45rem !important;
}

[data-testid="stChatInput"] button:hover {
    background: #0077ed !important;
}
            }

            [data-testid="stChatInput"] textarea {
                background: #ffffff !important;
                color: var(--ink) !important;
                font-size: 1rem !important;
            }

            [data-testid="stChatInput"] textarea::placeholder {
                color: var(--muted) !important;
            }

            /* Buttons */
            .stButton > button {
                background: #ffffff !important;
                color: var(--ink) !important;
                border: 1px solid var(--line) !important;
                border-radius: 10px !important;
                font-weight: 600;
            }

            .stButton > button:hover {
                color: var(--accent) !important;
                border-color: rgba(0, 113, 227, 0.45) !important;
            }

            /* Dropdown */
            [data-baseweb="select"] > div {
                background: #ffffff !important;
                color: var(--ink) !important;
                border-color: var(--line) !important;
            }

            .sidebar-note {
                color: var(--muted) !important;
                font-size: 0.84rem;
                line-height: 1.55;
                margin-top: 1rem;
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


def initialise_session():
    """Create message memory once for each browser session."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def clear_chat():
    """Clear conversation history."""
    st.session_state.messages = []


def get_gemini_history():
    """Convert stored messages into Gemini SDK format."""
    return [
        {
            "role": message["role"],
            "parts": [{"text": message["content"]}],
        }
        for message in st.session_state.messages
    ]


def generate_response(api_key, model_name, question):
    """Generate an answer using relevant document content when available."""
    context, sources = retrieve_context(question, api_key)

    client = genai.Client(api_key=api_key)

    if context:
        rag_instruction = f"""
        {SYSTEM_INSTRUCTION}

        Use the document context below to answer the user's question.

        Rules:
        - Answer using the document context.
        - If the answer is not in the document, clearly say so.
        - Do not invent information.

        Document context:
        {context}
        """
    else:
        rag_instruction = SYSTEM_INSTRUCTION

    response = client.models.generate_content(
        model=model_name,
        contents=get_gemini_history(),
        config={
            "system_instruction": rag_instruction,
        },
    )

    answer = response.text or "I could not generate a response."
    return answer, sources
# ---------------------------------------------------
# Interface
# ---------------------------------------------------
def render_sidebar(api_key):
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-logo">
                <span>✦</span>{APP_NAME}
            </div>
            <p class="sidebar-tagline">Your private AI workspace</p>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        model_name = st.selectbox(
            "Model",
            ["gemini-3.6-flash"],
            help="Fast Gemini model for chat, coding, and reasoning.",
        )

        st.divider()

        uploaded_file = st.file_uploader(
            "Add a document",
            type=["pdf", "txt", "docx", "csv", "xlsx", "doc"],
            help="Upload a document so NOVA can answer questions from it.",
        )

        if uploaded_file is not None:
            if st.button("Add document", use_container_width=True):
                if not api_key:
                    st.error("Your GEMINI_API_KEY was not found.")
                else:
                    try:
                        with st.spinner("Reading and indexing document..."):
                            chunk_count, message = index_document(
                                uploaded_file,
                                api_key,
                            )

                        if chunk_count > 0:
                            st.success(
                                f"Added {chunk_count} document sections."
                            )
                        else:
                            st.info(message)

                    except Exception as error:
                        st.error(f"Could not add this document: {error}")

        if st.button("Clear documents", use_container_width=True):
            clear_documents()
            st.success("All uploaded documents were removed.")

        st.divider()

        if st.button("Clear chat", use_container_width=True):
            clear_chat()
            st.rerun()

        st.markdown(
            """
            <p class="sidebar-note">
                Upload a document, click Add document, then ask NOVA questions
                about its contents.
            </p>
            """,
            unsafe_allow_html=True,
        )

    return model_name
def main():
    apply_custom_styles()
    initialise_session()

    api_key = get_api_key()
    selected_model = render_sidebar(api_key)

    st.markdown(
        f"<div class='nova-brand'><span>✦</span>{APP_NAME}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h1 class='nova-title'>Think it through.</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p class='nova-subtitle'>A calm space to ask, explore, and create.</p>",
        unsafe_allow_html=True,
    )

    # Show previous chat messages.
    for message in st.session_state.messages:
        display_role = "assistant" if message["role"] == "model" else "user"

        with st.chat_message(display_role):
            st.markdown(message["content"])

    # Receive a new user message.
    if prompt := st.chat_input(f"Message {APP_NAME}..."):
        if not api_key:
            st.error(
                "GEMINI_API_KEY was not found. Check your .env file and restart Streamlit."
            )
            st.stop()

        # Save and display the user's message.
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve PDF context and generate NOVA's answer.
        with st.chat_message("assistant"):
            with st.spinner(f"{APP_NAME} is thinking..."):
                try:
                    answer, sources = generate_response(
                        api_key,
                        selected_model,
                        prompt,
                    )
                except Exception as error:
                    st.error(f"Could not reach Gemini: {error}")
                    return

            st.markdown(answer)

            if sources:
                st.caption("Sources: " + " • ".join(sources))

        # Save Gemini's response in conversation memory.
        st.session_state.messages.append(
            {"role": "model", "content": answer}
        )


if __name__ == "__main__":
    main()