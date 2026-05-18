import streamlit as st
import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from query import generate_answer

# 🔹 Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# 🔹 Page config
st.set_page_config(page_title="College AI Assistant")

st.title("📚 College AI Assistant with PDF Upload")

# 🔹 Session memory
if "history" not in st.session_state:
    st.session_state.history = []

# 🔹 Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

# 🔹 Process PDF
def process_pdf(file):
    text = ""
    reader = PdfReader(file)

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# 🔹 Chunking
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# 🔹 Create FAISS index
def create_index(chunks):
    embeddings = embed_model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, "embeddings/index.faiss")

    with open("chunks.txt", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c + "\n---\n")


# 🔹 Button to process uploaded PDF
if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            text = process_pdf(uploaded_file)
            chunks = chunk_text(text)
            create_index(chunks)

        st.success("✅ PDF processed successfully!")

# 🔹 Ask question
query = st.text_input("Ask your question:")

if st.button("Get Answer"):
    if query:
        with st.spinner("🤖 Thinking..."):
            answer, sources = generate_answer(query)

        st.session_state.history.append((query, answer))

        st.subheader("🤖 Answer:")
        st.write(answer)

        st.subheader("📖 Sources:")
        for i, (idx, text) in enumerate(sources):
            st.info(text[:300] + "...")

    else:
        st.warning("Please enter a question")

# 🔹 Chat history
st.subheader("🧠 Chat History")

for q, a in st.session_state.history:
    st.write(f"**Q:** {q}")
    st.write(f"**A:** {a}")