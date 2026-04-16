import streamlit as st
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
import os

# -------------------- CONFIG --------------------
st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖", layout="wide")

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙️ Settings")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# -------------------- TITLE --------------------
st.title("🤖 AI PDF Assistant (RAG)")
st.markdown("Ask questions based on your uploaded PDFs")

# -------------------- FILE UPLOAD --------------------
uploaded_files = st.file_uploader("📄 Upload PDFs", type="pdf", accept_multiple_files=True)

# -------------------- INIT SESSION --------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- PROCESS PDFs --------------------
def process_pdfs(files):
    documents = []

    for file in files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        docs = loader.load()
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

# -------------------- RAG FUNCTION --------------------
def rag_answer(query, vectorstore, client):
    if not query or query.strip() == "":
        return "⚠️ Please enter a valid question.", []

    try:
        docs = vectorstore.similarity_search(query, k=3)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question using the context below.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        res = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = res.choices[0].message.content

        sources = [doc.metadata.get("source", "Unknown") for doc in docs]

        return answer, sources

    except Exception as e:
        return f"❌ Error: {str(e)}", []

# -------------------- BUILD VECTORSTORE --------------------
if uploaded_files:
    with st.spinner("📚 Processing PDFs..."):
        st.session_state.vectorstore = process_pdfs(uploaded_files)
    st.success("✅ PDFs processed successfully!")

# -------------------- CHAT INPUT --------------------
query = st.chat_input("Ask something about your PDF...")

# -------------------- CHAT LOGIC --------------------
if query:
    if not groq_api_key:
        st.error("❌ Please enter your Groq API key in sidebar.")
    elif st.session_state.vectorstore is None:
        st.error("❌ Please upload PDFs first.")
    else:
        client = Groq(api_key=groq_api_key)

        answer, sources = rag_answer(query, st.session_state.vectorstore, client)

        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("ai", answer, sources))

# -------------------- DISPLAY CHAT --------------------
for chat in st.session_state.chat_history:
    if chat[0] == "user":
        with st.chat_message("user"):
            st.markdown(chat[1])

    elif chat[0] == "ai":
        with st.chat_message("assistant"):
            st.markdown(chat[1])

            if len(chat) > 2 and chat[2]:
                with st.expander("📚 Sources"):
                    for src in chat[2]:
                        st.write(f"• {src}")git add .
git commit -m "update"
git push