"""
=================================================
  Smart Email Reply Agent — Groq API (Pure Python)
=================================================
No LangChain. No complex frameworks. Just Python + Groq.

SETUP:
  pip install groq

  Get your FREE Groq API key at: https://console.groq.com

  Set your API key (PowerShell):
    $env:GROQ_API_KEY = "your-key-here"
  OR just paste it when the program asks you.
"""

import os
from groq import Groq

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

MODEL = "llama-3.3-70b-versatile"   # Latest fast model on Groq (free tier)

TONE_OPTIONS = {
    "1": "polite and professional",
    "2": "friendly and warm",
    "3": "formal and strict",
    "4": "concise and direct",
    "5": "empathetic and supportive",
}


# ─────────────────────────────────────────────
#  INITIALIZE GROQ CLIENT
# ─────────────────────────────────────────────

def get_client() -> Groq:
    """
    Create and return a Groq client.
    Reads the API key from environment variable or asks the user.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("⚠️  GROQ_API_KEY not found in environment.")
        api_key = input("   Paste your Groq API key: ").strip()
    return Groq(api_key=api_key)


# ─────────────────────────────────────────────
#  FUNCTION 1 — ANALYZE EMAIL
# ─────────────────────────────────────────────

def analyze_email(client: Groq, email_text: str) -> str:
    """
    Analyzes the tone and intent of the given email.
    Returns a short structured analysis.
    """
    prompt = f"""You are an expert email analyst.

Analyze the following email and provide a SHORT structured analysis covering:
1. Tone       : (e.g., formal, informal, angry, polite, urgent, friendly)
2. Intent     : (e.g., request, complaint, inquiry, follow-up, appreciation)
3. Key Points : Bullet the 2-3 most important things the sender wants

Keep the analysis brief and clear. No fluff.

EMAIL:
\"\"\"
{email_text}
\"\"\"
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
#  FUNCTION 2 — GENERATE REPLY
# ─────────────────────────────────────────────

def generate_reply(client: Groq, email_text: str, tone: str) -> str:
    """
    Generates a professional email reply based on the original email and desired tone.
    """
    prompt = f"""You are a professional email assistant.

Write a complete email reply to the email below.

Rules:
- Tone must be: {tone}
- Address all points raised in the original email
- Keep it concise (3-5 sentences unless more detail is needed)
- Include a proper greeting and sign-off
- Do NOT add any explanation or commentary — just the email reply itself

ORIGINAL EMAIL:
\"\"\"
{email_text}
\"\"\"

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
#  HELPER — DISPLAY TONE MENU
# ─────────────────────────────────────────────

def choose_tone() -> str:
    """Displays tone options and returns the user's chosen tone string."""
    print("\n🎨 Choose reply tone:")
    for key, value in TONE_OPTIONS.items():
        print(f"   {key}. {value.title()}")
    while True:
        choice = input("\nEnter number (1-5): ").strip()
        if choice in TONE_OPTIONS:
            return TONE_OPTIONS[choice]
        print("   ❌ Invalid choice. Please enter 1-5.")


# ─────────────────────────────────────────────
#  MAIN — CLI INTERACTION LOOP
# ─────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   📧  Smart Email Reply Agent  |  Groq + Python  ║")
    print(f"║      Model : {MODEL:<36}║")
    print("╚══════════════════════════════════════════════════╝\n")

    try:
        client = get_client()
        print("✅ Connected to Groq!\n")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return

    print("─" * 52)
    print("Type 'exit' at any prompt to quit.\n")

    while True:
        print("📩 Paste the email you received (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line.lower() == "exit":
                print("\nGoodbye! 👋")
                return
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)

        email_text = "\n".join(lines).strip()
        if not email_text:
            print("⚠️  No email entered. Try again.\n")
            continue

        print("\n🔍 Analyzing email...\n")
        try:
            analysis = analyze_email(client, email_text)
            print("─" * 52)
            print("📊 EMAIL ANALYSIS")
            print("─" * 52)
            print(analysis)
            print("─" * 52)
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            continue

        tone = choose_tone()

        print(f"\n✍️  Generating {tone} reply...\n")
        try:
            reply = generate_reply(client, email_text, tone)
            print("─" * 52)
            print("📬 GENERATED REPLY")
            print("─" * 52)
            print(reply)
            print("─" * 52)
        except Exception as e:
            print(f"❌ Reply generation failed: {e}")
            continue

        again = input("\n🔁 Reply to another email? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye! 👋")
            break
        print()


if __name__ == "__main__":
    main()