import openai
import PyPDF2
import streamlit as st

# ==== Configure API Key ====
openai.api_key = "key"  # ← Replace with your actual API key

# ==== Read prompt from PDF ====
@st.cache_data
def extract_prompt_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        content = ""
        for page in reader.pages:
            content += page.extract_text()
    return content.strip()


# ==== Chat Function ====
def chat_with_gpt(user_input, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",  # You can also use gpt-3.5-turbo
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()


# ==== Streamlit UI Layout ====
def main():
    st.set_page_config(page_title="ADHD Communication Assistant", page_icon="🧠", layout="centered")
    st.title("🧠 ADHD Communication Assistant")
    st.write("Please enter your question in natural language. The AI will respond in a communication style designed for ADHD users.")

    # Load system prompt
    system_prompt = extract_prompt_from_pdf("C:/Users/87794/PycharmProjects/PythonProject1/ADHD.pdf")

    # User input
    user_input = st.text_area("✍️ Enter your question here", height=100)

    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please enter a question!")
        else:
            with st.spinner("AI is thinking... please wait."):
                response = chat_with_gpt(user_input, system_prompt)
            st.markdown("### 🤖 AI Response:")
            st.success(response)


if __name__ == "__main__":
    main()
