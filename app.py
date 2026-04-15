import streamlit as st
import streamlit.components.v1 as components
import faiss
import numpy as np
import os
import re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

# ================= SETUP =================

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("🤖 AI Assistant")

# ================= SESSION =================

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# ================= SIDEBAR =================

mode = st.sidebar.radio("Mode", ["Both", "PDF Only", "General AI Only"])

# ================= CORE =================

def extract_text(file):
    text = ""
    reader = PdfReader(file)
    for p in reader.pages:
        text += p.extract_text() or ""
    return text

def chunk_text(text, size=800, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

def create_index(chunks):
    emb = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(np.array(emb))
    return index

def retrieve(query, k=12):
    emb = embed_model.encode([query])
    D, I = st.session_state.index.search(np.array(emb), k)
    return [(st.session_state.chunks[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

# 🔥 RERANK
def rerank(results, query):
    q_words = set(query.lower().split())
    ranked = []
    for text, score in results:
        keyword_score = sum(1 for w in q_words if w in text.lower())
        final_score = keyword_score * 2 - score
        ranked.append((text, final_score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:5]

# 🔥 HIGHLIGHT
def highlight(text, query):
    for word in query.split():
        if len(word) > 3:
            text = re.sub(
                f"({word})",
                r"<span style='background-color:#FFD54F; color:black; font-weight:bold; padding:2px 4px; border-radius:4px;'>\1</span>",
                text,
                flags=re.IGNORECASE
            )
    return text

# 🔥 FORMAT CONTEXT
def format_context(ranked, query):
    formatted = []
    for i, (text, _) in enumerate(ranked):
        sentences = re.split(r'(?<=[.!?]) +', text)
        snippet = " ".join(sentences[:5])
        snippet = highlight(snippet, query)
        formatted.append((i+1, snippet, text))
    return formatted

# ================= LLM =================

def general_answer(query):
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": query}]
    )
    return res.choices[0].message.content

def rag_answer(query):
    results = retrieve(query)
    ranked = rerank(results, query)

    context_text = "\n\n".join([t for t, _ in ranked])

    prompt = f"""
Answer using ONLY the context below.
Also cite sources like [1], [2].

Context:
{context_text}

Question: {query}

Answer:
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content, ranked

# ================= UI =================

uploaded = st.file_uploader("📂 Upload PDF", type=["pdf"])

if uploaded and st.button("Process PDF"):
    text = extract_text(uploaded)
    chunks = chunk_text(text)
    st.session_state.index = create_index(chunks)
    st.session_state.chunks = chunks
    st.success("✅ PDF Ready!")

query = st.chat_input("Ask anything...")

if query:
    st.session_state.chat.append(("user", query))

    # 🤖 AI
    if mode in ["Both", "General AI Only"]:
        ans = general_answer(query)
        st.session_state.chat.append(("bot", (
            "🤖 AI GENERATED ANSWER",
            f"<div style='background:#1e1e2f; padding:15px; border-radius:10px; border-left:5px solid #4CAF50'>{ans}</div>"
        )))

    # 📄 PDF
    if mode in ["Both", "PDF Only"] and st.session_state.index:
        ans, ranked = rag_answer(query)
        sources = format_context(ranked, query)

        st.session_state.chat.append(("bot", (
            "📄 ANSWER FROM DOCUMENT",
            f"<div style='background:#2b1e1e; padding:15px; border-radius:10px; border-left:5px solid #FF9800'>{ans}</div>"
        )))

        st.session_state.chat.append(("sources", sources))

    if mode in ["PDF Only", "Both"] and st.session_state.index is None:
        st.session_state.chat.append(("bot", ("⚠️", "Upload PDF first")))

# ================= DISPLAY =================

for item in st.session_state.chat:

    role = item[0]

    if role == "user":
        with st.chat_message("user"):
            st.write(item[1])

    elif role == "bot":
        title, content = item[1]
        with st.chat_message("assistant"):
            st.markdown(f"### {title}")
            st.markdown(content, unsafe_allow_html=True)

    elif role == "sources":
        with st.chat_message("assistant"):
            st.markdown("## 📖 Detailed Sources")

            source_html = ""

            for idx, snippet, full_text in item[1]:
                source_html += f"""
                <details style='margin-bottom:10px; background:#0f172a; padding:10px; border-radius:10px'>
                    <summary style='cursor:pointer; font-weight:bold'>
                        📌 Source [{idx}] (Click to expand)
                    </summary>

                    <div style='padding:10px; margin-top:5px'>
                        {snippet}
                        <hr>
                        <small style='color:gray'>Full Context:</small><br>
                        {full_text}
                    </div>
                </details>
                """

            components.html(source_html, height=400, scrolling=True)