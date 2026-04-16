import faiss
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
ANSWER_MODES = (
    "PDF only",
    "AI only",
    "AI + PDF",
)

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


def get_groq_client(api_key):
    return Groq(api_key=api_key)


def format_pages(pages):
    if not pages:
        return "Page information unavailable"
    if len(pages) == 1:
        return f"Page {pages[0]}"
    return f"Pages {pages[0]}-{pages[-1]}"


def format_source_preview(text, max_length=260):
    preview = " ".join(text.split())
    if len(preview) <= max_length:
        return preview
    return preview[: max_length - 3].rstrip() + "..."


def should_show_sources(answer, sources):
    if not sources:
        return False

    normalized_answer = answer.lower()
    no_context_signals = [
        "i don't know",
        "i do not know",
        "not present in the context",
        "out of context",
        "no relevant pdf evidence",
        "no relevant pdf context",
        "cannot be determined from the uploaded pdf",
        "can't be determined from the uploaded pdf",
    ]
    return not any(signal in normalized_answer for signal in no_context_signals)


def generate_pdf_answer(query, api_key, index, chunk_records):
    if not query.strip():
        return "Please enter a valid question.", []

    retrieved_chunks = retrieve_relevant_chunks(query, index, chunk_records)
    if not retrieved_chunks:
        return "No relevant PDF content is available yet. Please upload and process a PDF first.", []

    context = "\n\n".join(chunk["content"] for chunk in retrieved_chunks)
    client = get_groq_client(api_key)

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


def generate_ai_answer(query, api_key):
    if not query.strip():
        return "Please enter a valid question.", []

    client = get_groq_client(api_key)
    prompt = f"""
You are a helpful AI assistant.
Answer the user's question using your general knowledge.
Write in a well-structured style:
- Start with a short introductory paragraph.
- Then use bullet points for the main ideas, steps, or facts when helpful.
- End with a short closing paragraph if clarification or context would help.
Avoid one-line or overly blunt answers.

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
        temperature=0.3,
    )

    answer = response.choices[0].message.content.strip()
    return answer, []


def generate_combined_answer(query, api_key, index, chunk_records):
    if not query.strip():
        return "Please enter a valid question.", []

    retrieved_chunks = retrieve_relevant_chunks(query, index, chunk_records)
    context = "\n\n".join(chunk["content"] for chunk in retrieved_chunks) if retrieved_chunks else ""
    client = get_groq_client(api_key)

    prompt = f"""
You are a helpful AI assistant.
Answer the user's question by combining:
1. Information from the uploaded PDFs when relevant.
2. Your own general knowledge when useful.

Be explicit when a point comes from the PDF context versus general AI knowledge.
If no PDF context is available, answer using general knowledge and mention that no PDF evidence was found.
Write the answer in a polished structure:
- Begin with a short summary paragraph.
- Use bullet points to separate PDF-based points from general AI knowledge where appropriate.
- Keep the tone clear, explanatory, and not overly brief.

PDF Context:
{context if context else "No relevant PDF context found."}

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
        temperature=0.3,
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

answer_mode = st.sidebar.radio("Answer mode", ANSWER_MODES)

st.title("AI PDF Assistant")
st.markdown("Upload one or more PDFs and choose whether answers come from the PDF, the AI model, or both.")

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
    else:
        try:
            if answer_mode == "PDF only":
                if st.session_state.vector_index is None:
                    answer = "Please upload and process at least one PDF first."
                    sources = []
                else:
                    answer, sources = generate_pdf_answer(
                        query,
                        groq_api_key,
                        st.session_state.vector_index,
                        st.session_state.chunk_records,
                    )
            elif answer_mode == "AI only":
                answer, sources = generate_ai_answer(query, groq_api_key)
            else:
                answer, sources = generate_combined_answer(
                    query,
                    groq_api_key,
                    st.session_state.vector_index,
                    st.session_state.chunk_records,
                )
        except Exception as exc:
            answer = f"Error while generating an answer: {exc}"
            sources = []

    st.session_state.chat_history.append(("assistant", f"Mode: {answer_mode}\n\n{answer}", sources))


for role, content, sources in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)

        if role == "assistant" and should_show_sources(content, sources):
            with st.expander(f"Sources and evidence ({len(sources)})"):
                st.caption("These PDF excerpts were retrieved to support the answer.")
                for index, item in enumerate(sources, start=1):
                    st.markdown(
                        f"""
**Source {index}: {item['source']}**

`{format_pages(item['pages'])}` | `{len(item['content'])} characters retrieved`

**Why this matters:** This is one of the most relevant PDF passages matched to your question.

**Preview:** {format_source_preview(item['content'])}
"""
                    )
                    if index != len(sources):
                        st.divider()
