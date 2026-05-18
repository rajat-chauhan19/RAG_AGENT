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