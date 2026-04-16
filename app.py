import faiss
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama3-70b-8192"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

load_dotenv()


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_configured_api_key():
    secret_key = st.secrets.get("GROQ_API_KEY", "")
    env_key = os.getenv("GROQ_API_KEY", "")
    return secret_key or env_key


def extract_text_from_pdfs(files):
    documents = []

    for file in files:
        reader = PdfReader(file)
        page_text = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                page_text.append((page_number, text))

        combined_text = "\n".join(text for _, text in page_text).strip()
        if combined_text:
            documents.append(
                {
                    "source": file.name,
                    "pages": [page for page, _ in page_text],
                    "text": combined_text,
                }
            )

    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max(1, chunk_size - overlap)

    return chunks


def build_vectorstore(documents):
    embedder = load_embedder()
    chunk_records = []

    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            chunk_records.append(
                {
                    "source": doc["source"],
                    "pages": doc["pages"],
                    "content": chunk,
                }
            )

    if not chunk_records:
        return None, []

    embeddings = embedder.encode(
        [record["content"] for record in chunk_records],
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, chunk_records


def retrieve_relevant_chunks(query, index, chunk_records, top_k=TOP_K):
    if index is None or not chunk_records:
        return []

    embedder = load_embedder()
    query_embedding = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = index.search(query_embedding, min(top_k, len(chunk_records)))

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunk_records):
            results.append(chunk_records[idx])

    return results


def generate_rag_answer(query, api_key, index, chunk_records):
    if not query.strip():
        return "Please enter a valid question.", []

    retrieved_chunks = retrieve_relevant_chunks(query, index, chunk_records)
    if not retrieved_chunks:
        return "No relevant PDF content is available yet. Please upload and process a PDF first.", []

    context = "\n\n".join(chunk["content"] for chunk in retrieved_chunks)
    client = Groq(api_key=api_key)

    prompt = f"""
You are a helpful PDF question-answering assistant.
Answer the user's question using only the provided context.
If the answer is not present in the context, say "I don't know based on the uploaded PDFs."

Context:
{context}

Question:
{query}
""".strip()

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    return answer, retrieved_chunks


st.set_page_config(page_title="RAG AI Assistant", page_icon="📄", layout="wide")

st.sidebar.title("Settings")
configured_api_key = get_configured_api_key()

if configured_api_key:
    groq_api_key = configured_api_key
    st.sidebar.success("Groq API key loaded automatically.")
else:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    st.sidebar.caption("Tip: set GROQ_API_KEY in .env or Streamlit secrets to preload it.")

st.title("AI PDF Assistant")
st.markdown("Upload one or more PDFs and ask questions grounded in their content.")

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

if "chunk_records" not in st.session_state:
    st.session_state.chunk_records = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if uploaded_files:
    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            documents = extract_text_from_pdfs(uploaded_files)

            if not documents:
                st.session_state.vector_index = None
                st.session_state.chunk_records = []
                st.warning("No readable text was found in the uploaded PDFs.")
            else:
                index, chunk_records = build_vectorstore(documents)
                st.session_state.vector_index = index
                st.session_state.chunk_records = chunk_records
                st.success("PDFs processed successfully.")


query = st.chat_input("Ask something about your PDFs...")

if query:
    st.session_state.chat_history.append(("user", query, None))

    if not groq_api_key:
        answer = "Please enter your Groq API key in the sidebar."
        sources = []
    elif st.session_state.vector_index is None:
        answer = "Please upload and process at least one PDF first."
        sources = []
    else:
        try:
            answer, sources = generate_rag_answer(
                query,
                groq_api_key,
                st.session_state.vector_index,
                st.session_state.chunk_records,
            )
        except Exception as exc:
            answer = f"Error while generating an answer: {exc}"
            sources = []

    st.session_state.chat_history.append(("assistant", answer, sources))


for role, content, sources in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)

        if role == "assistant" and sources:
            with st.expander("Sources"):
                for item in sources:
                    pages = ", ".join(str(page) for page in item["pages"][:5])
                    suffix = "..." if len(item["pages"]) > 5 else ""
                    st.write(f"- {item['source']} (pages: {pages}{suffix})")
