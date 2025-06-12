# hinglish_finance_rag/app.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["CHROMA_TELEMETRY"] = "False"

import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from dotenv import load_dotenv
import base64
import tempfile
from gtts import gTTS

# Load .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
base_url = "https://api.groq.com/openai/v1"
avatar_path = "static/avatar.jpeg"

# Load style prompt
with open("style_prompt.txt", "r", encoding="utf-8") as f:
    style_prompt = f.read()

# Load model (Groq + LLaMA3)
llm = ChatOpenAI(
    model_name="llama3-8b-8192",
    openai_api_base=base_url,
    openai_api_key=api_key,
    temperature=0.7,
    max_tokens=512
)

# Load Vector DB (switching from Chroma to FAISS for Streamlit compatibility)
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
import pickle
if os.path.exists("faiss_index.pkl"):
    with open("faiss_index.pkl", "rb") as f:
        vectordb = pickle.load(f)
else:
    from langchain.vectorstores import FAISS
    vectordb = FAISS.from_texts(["Welcome to the Hinglish Finance Chatbot!"], embedding)
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump(vectordb, f)
retriever = vectordb.as_retriever()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Prompt builder
def build_prompt(user_query, docs):
    context = "\n".join(d.page_content for d in docs)
    return f"""{style_prompt}

Context:
{context}

User: {user_query}
You:"""

# Generate answer
def generate_answer(user_query):
    docs = retriever.get_relevant_documents(user_query)
    prompt = build_prompt(user_query, docs)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else response

# Text to speech (optional)
def speak_text(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_bytes = open(fp.name, 'rb').read()
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f'<audio autoplay controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)

# UI Setup
st.set_page_config(page_title="🧠💬 Neeraj Arora", layout="centered")
st.title("🧠💬 Financial Advisor")
st.markdown("Hello dosto, I'm Neeraj Arora and you can ask me any question related to finance")

# 1️⃣ Show chat history first
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=avatar_path if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

# 2️⃣ Chat text input — always pinned to bottom
text_input = st.chat_input("Apka financial sawaal kya hai?")
if text_input:
    st.session_state.messages.append({"role": "user", "content": text_input})
    output = generate_answer(text_input)
    st.session_state.messages.append({"role": "assistant", "content": output})

    with st.chat_message("user"):
        st.markdown(text_input)
    with st.chat_message("assistant", avatar=avatar_path):
        st.markdown(output)
        # speak_text(output)
