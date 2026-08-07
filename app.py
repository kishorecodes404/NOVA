import streamlit as st
from dotenv import load_dotenv
import os
from google import genai
# Load the .env file
load_dotenv()

# Read the API key from .env
api_key = os.getenv("GEMINI_API_KEY")

# Create the Gemini client
client = genai.Client(api_key=api_key)
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI Chatbot")

st.write("Welcome! Ask me anything.")
user_question = st.text_input("Ask your question")
if user_question:
    st.write("You asked:", user_question)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=user_question
    )

    st.write("Gemini:", response.text)