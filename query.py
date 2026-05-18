import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 🔹 Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# 🔹 Lazy load LLM (prevents freezing)
generator = None

def get_generator():
    global generator
    if generator is None:
        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",  # can upgrade to flan-t5-large later
            device=-1
        )
    return generator


# 🔹 Load FAISS index
index = faiss.read_index("embeddings/index.faiss")


# 🔹 Load chunks
def load_chunks():
    with open("chunks.txt", "r", encoding="utf-8") as f:
        return f.read().split("\n---\n")

chunks = load_chunks()


# 🔍 Retrieve relevant chunks (increased k for better context)
def retrieve(query, k=5):
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k)

    results = [(i, chunks[i]) for i in I[0]]
    return results


# 🧠 Generate better answer
def generate_answer(query):
    gen = get_generator()

    results = retrieve(query)

    # Clean + structured context
    context = [r[1].strip() for r in results if r[1].strip()]
    combined_context = "\n".join(context)

    prompt = f"""
You are a helpful AI assistant.

Answer the question clearly, completely, and in proper sentences.
Do not give short answers. Explain properly using the context.

Context:
{combined_context}

Question:
{query}

Detailed Answer:
"""

    response = gen(
        prompt,
        max_length=512,
        do_sample=True,
        temperature=0.7
    )

    answer = response[0]['generated_text']

    return answer, results


# 🧪 CLI testing (with exit option)
if __name__ == "__main__":
    while True:
        q = input("\nAsk your question (type 'exit' to quit): ")

        if q.lower() == "exit":
            print("Exiting...")
            break

        answer, sources = generate_answer(q)

        print("\nAnswer:\n")
        print(answer)

        print("\nSources:\n")
        for i, (idx, text) in enumerate(sources):
            print(f"{i+1}. {text[:200]}...\n")