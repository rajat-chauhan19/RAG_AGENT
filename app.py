import streamlit as st
import streamlit.components.v1 as components
import faiss
import numpy as np
import os
import re
import time
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# ================= CONFIG =================
st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("📚 AI Assistant (RAG + Smart AI)")

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

def extract_best_sentences(text, query, max_sent=3):
    sentences = re.split(r'(?<=[.!?]) +', text)
    query_words = query.lower().split()

    scored = []
    for s in sentences:
        score = sum(1 for w in query_words if w in s.lower())
        if len(s.strip()) > 40:
            scored.append((s, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:max_sent]]

def highlight(text, query):
    for word in query.split():
        if len(word) > 3:
            text = re.sub(
                f"({word})",
                r"<span style='background:#FFD54F; color:black; font-weight:bold;'>\1</span>",
                text,
                flags=re.IGNORECASE
            )
    return text

def format_sources(results, query):
    formatted = []
    for i, (text, score) in enumerate(results):
        best_lines = extract_best_sentences(text, query)
        snippet = " ".join(best_lines)
        confidence = round(score * 100, 2)
        snippet = highlight(snippet, query)
        formatted.append((i+1, snippet, text, confidence))
    return formatted

def suggest_queries():
    if not st.session_state.chunks:
        return [
            "Upload a document first",
            "Ask questions related to the PDF",
            "Try switching to AI mode"
        ]

    chunks = st.session_state.chunks[:5]
    combined = " ".join(chunks)

    prompt = f"""
Generate 3 meaningful questions based on this document.

Document:
{combined}

Only return questions.
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = res.choices[0].message.content

    questions = []
    for line in raw.split("\n"):
        line = line.strip("-•1234567890. ")
        if len(line) > 5:
            questions.append(line)

    return questions[:3]

# ================= STREAMING =================

def stream_output(text):
    placeholder = st.empty()
    output = ""
    for word in text.split():
        output += word + " "
        placeholder.markdown(output)
        time.sleep(0.02)  # typing speed

# ================= AI =================

def rag_answer(query):
    results = retrieve(query)

    if not is_relevant(results):
        return None, None

    sources = format_sources(results, query)

    context = "\n\n".join([f"[{i}] {t}" for i, _, t, _ in sources])

    prompt = f"""
Answer ONLY from the context.
Use citations [1], [2].

Context:
{context}

Question: {query}
Answer:
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content, sources

def general_answer(query):
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": query}]
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
            st.session_state.chat.append(("pdf", "❌ Outside document scope"))
            st.session_state.chat.append(("suggestions", suggest_queries()))

    elif mode == "AI Only":
        st.session_state.chat.append(("ai", ai_answer))

# ================= DISPLAY =================

def render_sources(sources):
    html = ""
    for idx, snippet, full_text, conf in sources:
        html += f"""
        <div style='margin-bottom:20px; padding:15px; border-radius:12px; background:#0f172a; border-left:5px solid #38bdf8'>
            <div style='font-weight:bold'>
                📌 Source [{idx}] | Confidence: {conf}%
            </div>
            <div style='margin-top:8px'>
                {snippet}
            </div>
            <details>
                <summary style='cursor:pointer; color:#38bdf8'>Full Context</summary>
                <div style='margin-top:10px; color:#ccc'>
                    {full_text}
                </div>
            </details>
        </div>
        """
    components.html(html, height=600, scrolling=True)

for item in st.session_state.chat:

    if item[0] == "user":
        with st.chat_message("user"):
            st.write(item[1])

    elif item[0] == "pdf":
        with st.chat_message("assistant"):
            st.markdown("📄 **Answer from Document**")
            stream_output(item[1])

    elif item[0] == "ai":
        with st.chat_message("assistant"):
            st.markdown("🤖 **General AI Answer**")
            stream_output(item[1])

    elif item[0] == "sources":
        with st.chat_message("assistant"):
            st.markdown("📖 **Sources**")
            render_sources(item[1])

    elif item[0] == "suggestions":
        with st.chat_message("assistant"):
            st.markdown("💡 Suggested Questions")
            for q in item[1]:
                if st.button(q):
                    st.session_state.clicked_query = q