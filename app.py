import streamlit as st
from dotenv import load_dotenv
import os
from google import genai

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# ----------------------------
# Gemini Client
# ----------------------------
client = genai.Client(api_key=api_key)

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ----------------------------
# Session State
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("⚙️ Controls")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------
# Main UI
# ----------------------------
st.title("🤖 AI Chatbot")
st.caption("Powered by Google Gemini")

# ----------------------------
# Display Previous Messages
# ----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# Chat Input
# ----------------------------
user_question = st.chat_input("Ask me anything...")

# ----------------------------
# Generate Response
# ----------------------------
if user_question:

    # Store User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # Display User Message
    with st.chat_message("user"):
        st.markdown(user_question)

    # Build Conversation History
    prompt = ""

    for message in st.session_state.messages:
        prompt += f"{message['role']}: {message['content']}\n"

    # Gemini Response
    with st.spinner("🤖 Thinking..."):

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        ai_response = response.text.strip()

    # Save Assistant Message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )

    # Display Assistant Message
    with st.chat_message("assistant"):
        st.markdown(ai_response)