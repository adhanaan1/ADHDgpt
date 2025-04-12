# 🧠 ADHD Communication Assistant

This project provides two implementations of an AI assistant specifically designed to support communication with ADHD users.

## 📂 Project Structure

```
ADHD-Assistant/
├── ADHD.pdf                # Prompt source document
├── test2.py                # Simple prompt-based version using Streamlit
├── build_vector_store.py   # Builds FAISS index from ADHD.pdf using OpenAI embeddings
├── app.py                  # RAG-based assistant with Streamlit UI
├── faiss_index.index       # Generated FAISS index (after running build script)
├── chunks.pkl              # Stored text chunks from PDF
```

---

## 🚀 How to Run

### 1. Set up your environment
Make sure you have Python 3.8+ and install the required dependencies:

```bash
pip install openai faiss-cpu PyPDF2 streamlit numpy
```

### 2. Generate the vector store (for RAG version)
Run the following to extract text from `ADHD.pdf`, create embeddings, and build a FAISS index:

```bash
python build_vector_store.py
```

### 3. Run the RAG-based assistant
Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

### 4. (Optional) Run the simpler prompt-based version
This version uses a single system prompt extracted from `ADHD.pdf`:

```bash
streamlit run test2.py
```

---

## ✨ About the Approaches

- **test2.py**: A basic version that loads a static prompt from a PDF and sends user input to the OpenAI API, with replies tailored for ADHD communication style.

- **build_vector_store.py + app.py**: A Retrieval-Augmented Generation (RAG) implementation. It dynamically retrieves the most relevant content from the PDF to use as context in each response, providing more accurate and document-grounded replies.

---

## 🤝 Contributing

Feel free to modify, build upon, or extend this project! Feedback and improvements are always welcome.

---

## 📬 Contact

For any questions or suggestions, feel free to reach out!

---

**Enjoy building assistants that truly support ADHD users!**

