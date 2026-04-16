import streamlit as st
import streamlit.components.v1 as components
import faiss
import numpy as np
import os
import re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# ================= CONFIG =================
st.set_page_config(page_title="RAGenius AI", page_icon="🤖", layout="wide")

# ================= CUSTOM UI =================
st.markdown("""
<style>
.chat-card {
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}
.user-card {
    background-color: #1e293b;
    border-left: 5px solid #38bdf8;
}
.ai-card {
    background-color: #0f172a;
    border-left: 5px solid #4CAF50;
}
.pdf-card {
    background-color: #0f172a;
    border-left: 5px solid orange;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 RAGenius AI")
st.caption("Smart PDF + AI Assistant")

# ================= API =================
API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ================= SESSION =================
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.chat = []
    st.session_state.clicked_query = None

# ================= MODE =================
mode = st.selectbox("Select Answer Mode", ["Both", "PDF Only", "AI Only"])

# ================= FUNCTIONS =================

def extract_text(file):
    reader = PdfReader(file)
    return "".join([p.extract_text() or "" for p in reader.pages])

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

def retrieve(query, k=8):
    emb = embed_model.encode([query])
    D, I = st.session_state.index.search(np.array(emb), k)

    results = []
    for idx, i in enumerate(I[0]):
        distance = float(D[0][idx])
        similarity = 1 / (1 + distance)
        results.append((st.session_state.chunks[i], similarity))

    return results

def is_relevant(results, threshold=0.35):
    avg = sum(score for _, score in results) / len(results)
    return avg > threshold

def format_sources(results):
    formatted = []
    for i, (text, score) in enumerate(results):
        confidence = round(score * 100, 2)
        formatted.append((i+1, text[:300], confidence))
    return formatted

# ================= SUGGESTIONS =================

def suggest_queries():
    if not st.session_state.chunks:
        return ["Upload a PDF first", "Ask from document", "Switch to AI mode"]

    return [
        "Explain main concept of document",
        "Give summary of document",
        "What are key points?"
    ]

# ================= AI =================

def rag_answer(query):
    results = retrieve(query)

    if not is_relevant(results):
        return None, None

    sources = format_sources(results)
    context = "\n\n".join([f"[{i}] {t}" for i, t, _ in sources])

    prompt = f"""
Answer ONLY from context.

### 📘 Definition
...

### 🔑 Key Points
- ...

### 📌 Explanation
...

Use citations [1], [2].

Context:
{context}

Question: {query}
Answer:
"""

    res = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content, sources

def general_answer(query):
    prompt = f"""
Answer in structured format:

### 📘 Definition
...

### 🔑 Key Points
- ...

### 📌 Explanation
...

### 🌍 Examples
- ...

Question: {query}
Answer:
"""

    res = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content

# ================= UI =================

uploaded = st.file_uploader("📂 Upload PDF", type=["pdf"])

if uploaded and st.button("Process PDF"):
    text = extract_text(uploaded)
    chunks = chunk_text(text)
    st.session_state.index = create_index(chunks)
    st.session_state.chunks = chunks
    st.success("✅ PDF processed")

query = st.chat_input("Ask anything...")

if st.session_state.clicked_query:
    query = st.session_state.clicked_query
    st.session_state.clicked_query = None

if query:
    st.session_state.chat.append(("user", query))

    has_pdf = st.session_state.index is not None

    pdf_answer, sources = (None, None)
    ai_answer = None

    if has_pdf:
        pdf_answer, sources = rag_answer(query)

    if mode in ["Both", "AI Only"] or (mode == "PDF Only" and pdf_answer is None):
        ai_answer = general_answer(query)

    if mode == "Both":
        if pdf_answer:
            st.session_state.chat.append(("pdf", pdf_answer))
            st.session_state.chat.append(("sources", sources))
        if ai_answer:
            st.session_state.chat.append(("ai", ai_answer))

    elif mode == "PDF Only":
        if pdf_answer:
            st.session_state.chat.append(("pdf", pdf_answer))
            st.session_state.chat.append(("sources", sources))
        else:
            st.session_state.chat.append(("pdf", "❌ Out of document scope"))
            st.session_state.chat.append(("suggestions", suggest_queries()))

    elif mode == "AI Only":
        st.session_state.chat.append(("ai", ai_answer))

# ================= DISPLAY =================

for item in st.session_state.chat:

    if item[0] == "user":
        st.markdown(f"<div class='chat-card user-card'>👤 {item[1]}</div>", unsafe_allow_html=True)

    elif item[0] == "pdf":
        st.markdown(f"<div class='chat-card pdf-card'>📄 {item[1]}</div>", unsafe_allow_html=True)

    elif item[0] == "ai":
        st.markdown(f"<div class='chat-card ai-card'>🤖 {item[1]}</div>", unsafe_allow_html=True)

    elif item[0] == "sources":
        st.markdown("### 📖 Sources")
        for idx, text, conf in item[1]:
            st.markdown(f"**[{idx}] ({conf}% confidence)** {text}")

    elif item[0] == "suggestions":
        st.markdown("### 💡 Suggestions")
        for q in item[1]:
            if st.button(q):
                st.session_state.clicked_query = q