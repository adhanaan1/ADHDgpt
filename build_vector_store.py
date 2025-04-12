import os

import openai
import PyPDF2
import numpy as np
import faiss
import pickle
import re

# Set OpenAI API key
openai.api_key = "your-key"
# ==== Clean text: remove special characters, keep English, Chinese, and basic punctuation ====
def clean_text(text):
    # Keep English and common punctuation, remove control characters
    return re.sub(r'[^\x00-\x7F\s.,!?\'\"()-]', ' ', text)

# ==== Get embedding vector ====
def get_embedding(text, engine="text-embedding-ada-002"):
    text = clean_text(text.replace("\n", " "))
    return openai.Embedding.create(
        input=[text],
        model=engine
    )["data"][0]["embedding"]

# ==== Extract text from PDF ====
def extract_text_from_pdf(path):
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# ==== Split text into chunks ====
def split_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# ==== Build FAISS index ====
def build_faiss_index(chunks):
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"\n📦 Processing chunk {i+1} (length: {len(chunk)} characters):")
        print(f"Preview: {chunk[:100]}...")
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

# ==== Main ====
if __name__ == "__main__":
    pdf_path = "ADHD.pdf"  # Modify this path if needed
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)

    print(f"\n📄 Number of text chunks extracted: {len(chunks)}")
    print("🔍 Generating embeddings and building FAISS index...\n")

    index, embeddings = build_faiss_index(chunks)

    # Save FAISS index and original chunks
    faiss.write_index(index, "faiss_index.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("\n✅ Vector store built successfully! Saved files: faiss_index.index and chunks.pkl")
