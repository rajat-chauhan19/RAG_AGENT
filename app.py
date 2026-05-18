import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
load_dotenv()

MODEL = "llama-3.3-70b-versatile"

TONE_OPTIONS = {
    "Polite and Professional": "polite and professional",
    "Friendly and Warm": "friendly and warm",
    "Formal and Strict": "formal and strict",
    "Concise and Direct": "concise and direct",
    "Empathetic and Supportive": "empathetic and supportive",
}

REPLY_TYPE_OPTIONS = {
    "Accept / Agree": "accepting and agreeing with the sender's request or proposal",
    "Decline / Reject": "politely declining or rejecting the sender's request",
    "Ask for Clarification": "asking for more details or clarification before proceeding",
    "Provide Information": "providing the requested information or update",
    "Apologize": "apologizing for the issue or delay mentioned",
    "Follow Up": "following up on a previous conversation or pending matter",
    "Acknowledge & Defer": "acknowledging the email but deferring the response to a later time",
    "Escalate": "escalating the matter to a higher authority or another team",
}

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in .env file.")
        st.stop()
    return Groq(api_key=api_key)


def analyze_email(client: Groq, email_text: str) -> str:
    prompt = f"""
You are an expert email analyst.

Analyze the following email and provide a SHORT structured analysis covering:
1. Tone       : (e.g., formal, informal, angry, polite, urgent, friendly)
2. Intent     : (e.g., request, complaint, inquiry, follow-up, appreciation)
3. Key Points : Bullet the 2-3 most important things the sender wants

Keep the analysis brief and clear. No fluff.

EMAIL:
\"\"\"{email_text}\"\"\"
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def generate_reply(client: Groq, email_text: str, tone: str, reply_type: str) -> str:
    prompt = f"""
You are a professional email assistant.

Write a complete email reply to the email below.

Rules:
- Tone must be: {tone}
- Reply type / intent: {reply_type}
- Your reply should clearly reflect the reply type above — this is the PURPOSE of your response
- Address all relevant points raised in the original email
- Keep it concise (3–5 sentences unless more detail is needed)
- Include a proper greeting and sign-off
- Do NOT add any explanation or commentary — just the email reply itself

ORIGINAL EMAIL:
\"\"\"{email_text}\"\"\"

Write the reply now:
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
#  ANALYSIS CARD RENDERER
# ─────────────────────────────────────────────
def render_analysis(analysis_text: str):
    lines = analysis_text.strip().split('\n')
    tone = intent = ""
    key_points = []

    for line in lines:
        clean = line.strip().replace("**", "").strip()

        if ":" in clean:
            key = clean.split(":", 1)[0].strip().lower()
            value = clean.split(":", 1)[1].strip()

            if "tone" in key:
                tone = value
            elif "intent" in key:
                intent = value
            elif "key point" in key or "key_point" in key:
                if value:
                    key_points.append(value)

        stripped = line.strip().replace("**", "")
        if stripped.startswith(("* ", "- ", "• ")):
            point = stripped.lstrip("*-•").strip()
            if point and ":" not in point and len(point) > 5:
                key_points.append(point)

    st.markdown("""
    <style>
    .analysis-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 16px;
        padding: 28px 32px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        font-family: 'Segoe UI', sans-serif;
    }
    .analysis-title {
        font-size: 18px;
        font-weight: 700;
        color: #63b3ed;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 20px;
    }
    .metric-row {
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
    }
    .metric-box {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px 18px;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 15px;
        font-weight: 600;
        color: #e2e8f0;
    }
    .tone-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(99,179,237,0.15), rgba(154,117,255,0.15));
        border: 1px solid rgba(99,179,237,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 13px;
        color: #90cdf4;
        font-weight: 500;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .keypoints-title {
        font-size: 11px;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 12px;
    }
    .keypoint-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 10px 14px;
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #63b3ed;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
        color: #cbd5e0;
        font-size: 14px;
        line-height: 1.5;
    }
    .dot {
        width: 6px;
        height: 6px;
        min-width: 6px;
        background: #63b3ed;
        border-radius: 50%;
        margin-top: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

    tone_badges = "".join(
        f'<span class="tone-badge">{t.strip()}</span>'
        for t in tone.split(",")
    ) if tone else "<span class='tone-badge'>N/A</span>"

    points_html = "".join(
        f'<div class="keypoint-item"><div class="dot"></div><span>{p}</span></div>'
        for p in key_points
    ) if key_points else "<div class='keypoint-item'><div class='dot'></div><span>No key points extracted.</span></div>"

    st.markdown(f"""
    <div class="analysis-card">
        <div class="analysis-title">📊 Email Analysis</div>
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-label">🎭 Tone</div>
                <div class="metric-value">{tone_badges}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">🎯 Intent</div>
                <div class="metric-value">{intent or "N/A"}</div>
            </div>
        </div>
        <div class="keypoints-title">📌 Key Points</div>
        {points_html}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  STREAMLIT APP
# ─────────────────────────────────────────────
st.set_page_config(page_title="Smart Email Reply Agent", page_icon="📧", layout="centered")

st.title("📧 Smart Email Reply Agent — Groq + Streamlit")
st.caption("Powered by **Groq API** and **Python** — no complex frameworks.")

st.divider()

# Email input
email_text = st.text_area("📩 Paste the email content here", height=200, placeholder="Enter an email message...")

# Two columns for tone and reply type
col1, col2 = st.columns(2)

with col1:
    tone_label = st.selectbox("🎨 Reply Tone", list(TONE_OPTIONS.keys()))
    tone = TONE_OPTIONS[tone_label]

with col2:
    reply_type_label = st.selectbox("📨 Reply Type", list(REPLY_TYPE_OPTIONS.keys()))
    reply_type = REPLY_TYPE_OPTIONS[reply_type_label]

# Show selected combination as a hint
st.caption(f"💡 Will generate a **{reply_type_label}** reply in a **{tone_label}** tone.")

# Action button
if st.button("Analyze and Generate Reply", use_container_width=True):
    if not email_text.strip():
        st.warning("Please paste an email to analyze.")
    else:
        try:
            client = get_client()

            with st.spinner("Analyzing email..."):
                analysis = analyze_email(client, email_text)
            render_analysis(analysis)

            with st.spinner(f"Generating {reply_type_label.lower()} reply in {tone_label.lower()} tone..."):
                reply = generate_reply(client, email_text, tone, reply_type)

            st.subheader("📬 Generated Reply")
            st.text_area("Your AI-generated reply:", reply, height=200)

        except Exception as e:
            st.error(f"❌ Error: {e}")

st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Made with ❤️ using Groq + Streamlit</p>",
    unsafe_allow_html=True
)
=======
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
        "pdf-based answer: no relevant information found in the uploaded pdfs",
    ]
    return not any(signal in normalized_answer for signal in no_context_signals)


def combined_answer_has_pdf_evidence(answer):
    normalized_answer = answer.lower()
    return "pdf-based answer" in normalized_answer and (
        "no relevant information found in the uploaded pdfs" not in normalized_answer
    )


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
Answer the user's question in exactly two sections with these headings:
1. PDF-based answer
2. AI-based answer

Rules:
- In the PDF-based answer section, use only the uploaded PDF context.
- If the PDF context does not support the answer, write exactly: "No relevant information found in the uploaded PDFs."
- In the AI-based answer section, answer using general knowledge in a clear and well-structured way.
- Keep the AI-based answer readable with a short paragraph followed by bullet points when helpful.
- Do not merge the two sections together.

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


st.set_page_config(page_title="RAG AI Assistant", page_icon="📚", layout="wide")

st.sidebar.title("⚙️ Settings")
configured_api_key = get_configured_api_key()

if configured_api_key:
    groq_api_key = configured_api_key
    st.sidebar.success("🔐 Groq API key loaded automatically.")
else:
    groq_api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password")
    st.sidebar.caption("💡 Tip: set GROQ_API_KEY in `.env` or Streamlit secrets to preload it.")

answer_mode = st.sidebar.radio("🧠 Answer mode", ANSWER_MODES)

st.title("🤖 AI PDF Assistant")
st.markdown("### ✨ Turn your PDFs into a smart research companion with grounded answers, AI insights, or both together.")
st.info(f"📌 Current answer mode: **{answer_mode}**")

uploaded_files = st.file_uploader("📄 Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

if "chunk_records" not in st.session_state:
    st.session_state.chunk_records = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if uploaded_files:
    if st.button("🚀 Process PDFs"):
        with st.spinner("📚 Processing PDFs..."):
            documents = extract_text_from_pdfs(uploaded_files)

            if not documents:
                st.session_state.vector_index = None
                st.session_state.chunk_records = []
                st.warning("⚠️ No readable text was found in the uploaded PDFs.")
            else:
                index, chunk_records = build_vectorstore(documents)
                st.session_state.vector_index = index
                st.session_state.chunk_records = chunk_records
                st.success("✅ PDFs processed successfully.")


query = st.chat_input("💬 Ask something about your PDFs...")

if query:
    st.session_state.chat_history.append(("user", query, None))

    if not groq_api_key:
        answer = "🔑 Please enter your Groq API key in the sidebar."
        sources = []
    else:
        try:
            if answer_mode == "PDF only":
                if st.session_state.vector_index is None:
                    answer = "📄 Please upload and process at least one PDF first."
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
            answer = f"❌ Error while generating an answer: {exc}"
            sources = []

    st.session_state.chat_history.append(("assistant", f"📌 Mode selected: {answer_mode}\n\n{answer}", sources))


for role, content, sources in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)

        show_sources = role == "assistant" and should_show_sources(content, sources)
        if show_sources and "Mode selected: AI + PDF" in content:
            show_sources = combined_answer_has_pdf_evidence(content)

        if show_sources:
            with st.expander(f"📚 Sources and evidence ({len(sources)})"):
                st.caption("📝 These PDF excerpts were retrieved to support the answer.")
                for index, item in enumerate(sources, start=1):
                    st.markdown(
                        f"""
**📎 Source {index}: {item['source']}**

`{format_pages(item['pages'])}` | `{len(item['content'])} characters retrieved`

**🔍 Why this matters:** This is one of the most relevant PDF passages matched to your question.

**👀 Preview:** {format_source_preview(item['content'])}
"""
                    )
                    if index != len(sources):
                        st.divider()
>>>>>>> ac96c5687dc0bd79970c94caaba91233de844a45
