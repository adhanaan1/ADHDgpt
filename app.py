import os
import streamlit as st
import openai
import faiss
import pickle
import numpy as np
import re
import markdown2
import random
import uuid
import hashlib

# Set OpenAI API key
openai.api_key = "your-key"

# Utility functions
def clean_text(text):
    return re.sub(r"[^\x00-\x7F\u4e00-\u9fff。，？“”‘’《》\(\)（）【】·—：；\-_\[\]{}.,!?\"'\s]", ' ', text)

def get_embedding(text, engine="text-embedding-ada-002"):
    text = clean_text(text.replace("\n", " ").strip())
    embedding_response = openai.Embedding.create(input=[text], model=engine)
    return embedding_response["data"][0]["embedding"]

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("faiss_index.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve_context(query, index, chunks, top_k=3):
    query_vec = get_embedding(query)
    D, I = index.search(np.array([query_vec]).astype("float32"), top_k)
    return "\n\n".join([chunks[i] for i in I[0]])

def extract_suggested_questions(text):
    pattern = r"(?m)^\s*1\.\s+(.*?)\s*$\n^\s*2\.\s+(.*?)\s*$\n^\s*3\.\s+(.*?)\s*$"
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return []
    return [match.group(1).strip(), match.group(2).strip(), match.group(3).strip()]

def summarize_text(text):
    prompt = f"Summarize the following in 2-3 sentences:\n\n{text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.5
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Failed to summarize: {e}"

def get_deep_explanation(text):
    prompt = f"Give a deep, technical, and detailed explanation of the following topic. Be thorough and structured, but still ADHD-friendly:\n\n{text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Failed to expand: {e}"

def format_response_with_boxes(response):
    import hashlib

    paragraphs = response.split('\n\n')
    if len(paragraphs) > 3:
        paragraphs = paragraphs[:-3]

    fixed_colors = ['#e0f7fa', '#f1f8e9', '#fce4ec', '#fff9c4', '#e8f5e9', '#f3e5f5', '#f9fbe7']
    color_index = 0

    for idx, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        word_count = len(paragraph.split())
        if word_count < 20:
            continue
        paragraph_html = markdown2.markdown(paragraph).strip()

        if paragraph.startswith('#') or paragraph_html.startswith('<h'):
            st.markdown(paragraph_html, unsafe_allow_html=True)
            continue

        if paragraph_html.startswith('<p>') and paragraph_html.endswith('</p>'):
            paragraph_html = paragraph_html[3:-4]

        stable_id = hashlib.md5(paragraph.encode()).hexdigest()[:8]
        base_key = f"tldr_{idx}_{stable_id}"
        summary_key = f"summary_{base_key}"
        expand_key = f"expand_{base_key}"

        background = fixed_colors[min(color_index, len(fixed_colors) - 1)]
        color_index += 1

        with st.container():
            if idx == 0:
                st.markdown(paragraph_html, unsafe_allow_html=True)
                continue

            html_start = f"""
                <div style="border:1px solid #66bb6a; padding:12px 16px 8px 16px; border-radius:10px;
                            margin:8px auto; width: 100%; max-width: 900px;
                            background-color:{background};">
                    <div style="margin-bottom: 12px;">{paragraph_html}</div>
            """
            st.markdown(html_start, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])

            with col1:
                if st.button("TL;DR", key=base_key):
                    summary = summarize_text(paragraph)
                    st.session_state[summary_key] = summary

            expand_text_key = f"expand_text_{base_key}" 
            with col2:
                if st.button("Expand", key=expand_key):  # this key is only for the button widget
                    expanded = get_deep_explanation(paragraph)
                    st.session_state[expand_text_key] = expanded  # this key stores the actual text

            if expand_text_key in st.session_state and isinstance(st.session_state[expand_text_key], str):
                st.markdown(
                    f'<div style="border:1px solid #8e24aa; padding:10px 14px; border-radius:8px; '
                    f'margin:8px auto; width: 90%; max-width: 850px; text-align: left; background-color:#f3e5f5;">'
                    f'<strong>Deep Dive:</strong><br>{markdown2.markdown(st.session_state[expand_text_key])}</div>',
                    unsafe_allow_html=True,
                )
 # This ensures we're saving the actual text


            if summary_key in st.session_state:
                st.markdown(
                    f'<div style="border:1px solid #42a5f5; padding:10px 14px; border-radius:8px; '
                    f'margin:8px auto; width: 90%; max-width: 850px; text-align: left; background-color:#e3f2fd;">'
                    f'<strong>Summary:</strong> {st.session_state[summary_key]}</div>',
                    unsafe_allow_html=True,
                )


            st.markdown("</div>", unsafe_allow_html=True)

def chat_with_bot(user_input, conversation_history, context=""):
    focus = st.session_state.get("focus_mode", False)

    # Get the previous assistant message as the current topic
    current_topic = ""
    for msg in reversed(conversation_history):
        if msg["role"] == "assistant":
            current_topic = msg["content"]
            break

    # Check if the input is off-topic
    if focus and current_topic:
        check_prompt = (
            f"The user asked: '{user_input}'\n"
            f"The current topic is: '{current_topic}'\n\n"
            "Is the user going off-topic? Just answer Yes or No."
        )

        off_topic_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": check_prompt}],
            temperature=0
        )

        off_topic_reply = off_topic_response["choices"][0]["message"]["content"].strip().lower()

        if off_topic_reply.startswith("yes"):
            redirect_msg = (
                "Let’s stay focused on the current topic for a deeper understanding. "
                "Once we’ve explored this fully, we can jump to your next curiosity.\n\n"
                f"Try rephrasing your question to relate more closely to what we’re discussing."
            )
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": redirect_msg})
            return conversation_history, redirect_msg

    # === Your original chat logic starts here ===

    system_prompt = (
        "# CoSTAR Prompt\n"
        "## C - Context:\n"
        "You are a chatbot designed to support individuals with ADHD. Your tone should be friendly, structured, and clear. Provide easy to understand detail and complite answers and give fun examples. don't make the paragraphs too short. don't intract with the user like child and don't avoid tecnical topics just delve into it smoothly start from basic and go to advanced in each response\n"
        "\n"
        "## o - Objective:\n"
        "Help users understand technical concepts in a fun and engaging way.\n"
        "\n"
        "## S - Style:\n"
        "Use bullet points, avoid jargon, and maintain a warm, non-judgmental, fun tone.\n"
        "\n"
        "## T - Task:\n"
        "Respond to user queries by providing detailed responses.\n"
        "\n"
        "## A - Audience:\n"
        "Primarily individuals with ADHD trying to understand technical concepts. They may struggle with focus, planning, and emotional regulation.\n"
        "\n"
        "## R - Response Format:\n"
        "- Use bullet points for clarity\n"
        "- Include examples when possible\n"
        "- Use emojis when appropriate\n"
        "- Always include positive reinforcement\n"
        "- End each response with 3 follow-up topics (3 to 5 words each max no extra explaintion) in the following format:\n"
        "  1. Follow-up option one(3 to 5 words)\n"
        "  2. Follow-up option two(3 to 5 words)\n"
        "  3. Follow-up option three(3 to 5 words)"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history
    ]

    full_input = f"Context:\n{context}\n\nQuestion: {user_input}"
    messages.append({"role": "user", "content": full_input})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=1500 
    )

    bot_response = response['choices'][0]['message']['content']

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": bot_response})

    return conversation_history, bot_response


def generate_quiz_questions():
    # Gather all assistant responses in the conversation
    full_topic = ""
    for msg in st.session_state.conversation_history:
        if msg["role"] == "assistant":
            full_topic += msg["content"] + "\n\n"

    if not full_topic.strip():
        return []

    quiz_prompt = (
        "You are an expert technical instructor. Based on the following explanations, "
        "generate 10 multiple choice questions to test understanding of the concepts covered.\n\n"
        "Only use what was explained — do not add unrelated content.\n\n"
        "Each question should include:\n"
        "- 'question': a clear technical question\n"
        "- 'options': 4 realistic options (1 correct + 3 plausible distractors)\n"
        "- 'answer': the correct option (text, not letter)\n\n"
        "Format your output as a JSON list of dictionaries.\n\n"
        "Here is the content to base the questions on:\n\n"
        f"{full_topic}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": quiz_prompt}],
            temperature=0.5,
            max_tokens=1200
        )
        import json
        return json.loads(response['choices'][0]['message']['content'])
    except Exception as e:
        st.error(f"Failed to generate quiz: {e}")
        return []


def main():
    st.set_page_config(page_title="ADHD friendly chatbot", page_icon="")
    st.markdown("""
        <style>
        .block-container {
            padding-bottom: 120px !important;
        }
        .stForm {
            position: fixed;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100%;
            max-width: 720px;
            background-color: white;
            padding: 1rem;
            box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.1);
            z-index: 999;
        }
        .spinner-container {
            text-align: center;
            padding: 5px;
            font-weight: bold;
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧠ADHD friendly chatbot")

    # --- Session setup ---
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'suggested_questions' not in st.session_state:
        st.session_state.suggested_questions = []
    if 'focus_mode' not in st.session_state:
        st.session_state.focus_mode = True
    if 'unlock_requested' not in st.session_state:
        st.session_state.unlock_requested = False
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = []
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    
    # --- Focus Mode Lock Logic ---
    # Focus Mode toggle logic
    if st.session_state.conversation_history:
        if not st.session_state.focus_mode:
            if st.session_state.show_focus_toggle:
                st.session_state.focus_mode = st.toggle("Focus Mode (stay on topic)", value=False)
        else:
            if st.button("Request to turn off Focus Mode"):
                st.session_state.unlock_requested = True
    else:
        st.session_state.focus_mode = st.toggle("Focus Mode (stay on topic)", value=True)

    if st.session_state.unlock_requested and not st.session_state.quiz_started:
        st.markdown("### Do you want to turn off Focus Mode?")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Yes, quiz me"):
                st.session_state.quiz_started = True
                st.session_state.quiz_score = 0
                st.session_state.quiz_questions = generate_quiz_questions()
        with col2:
            if st.button("No, stay focused"):
                st.session_state.unlock_requested = False

    if st.session_state.quiz_started and st.session_state.quiz_questions:
        st.markdown("### Answer these to turn off Focus Mode:")

        correct_count = 0
        total = len(st.session_state.quiz_questions)

        for i, q in enumerate(st.session_state.quiz_questions):
            user_answer = st.radio(
                f"{i+1}. {q['question']}",
                q['options'],
                key=f"quiz_q_{i}"
            )
            correct = q['answer'].strip().lower()
            user = user_answer.strip().lower()
            if user == correct:
                correct_count += 1


        if st.button("Submit answers"):
            score_percent = (correct_count / total) * 100
            st.session_state.quiz_score = score_percent
            st.success(f"You scored {score_percent:.0f}%")

            if score_percent >= 70:
                st.success("Great! You’ve unlocked the ability to turn off Focus Mode.")
                st.session_state.focus_mode = False

        # Clear everything related to quiz
                st.session_state.quiz_started = False
                st.session_state.unlock_requested = False
                st.session_state.quiz_questions = []
                st.session_state.quiz_score = 0

        # Show the focus toggle again
                st.session_state.show_focus_toggle = True

                st.rerun()  # force UI refresh to hide quiz instantly
            else:
                st.warning("Try again! You need at least 70% to turn off Focus Mode.")
    if 'show_focus_toggle' not in st.session_state:
        st.session_state.show_focus_toggle = True
    # --- Load vector store ---
    index, chunks = load_faiss_index()

    # --- Show processing spinner ---
    if 'processing' in st.session_state and st.session_state['processing']:
        st.markdown('<div class="spinner-container">Thinking...</div>', unsafe_allow_html=True)

    # --- Show chat history ---
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.markdown(
                f"<div style='text-align: right; background-color:#eeeeee; padding:10px; border-radius:10px; margin-bottom:10px; margin-top:20px; display: inline-block; float: right; clear: both; max-width: 700px;'>{message['content']}</div>",
                unsafe_allow_html=True)
        else:
            format_response_with_boxes(message["content"])

    # --- Suggested follow-ups ---
    if st.session_state.suggested_questions:
        st.markdown("do you want to know about:")
        for i, question in enumerate(st.session_state.suggested_questions):
            full_text = f"I want to know about {question}"
            if st.button(question, key=f"suggestion_button_{i}"):
                context = retrieve_context(full_text, index, chunks)
                st.session_state.conversation_history, new_response = chat_with_bot(full_text, st.session_state.conversation_history, context)
                st.session_state.suggested_questions = extract_suggested_questions(new_response)
                st.rerun()

    # --- Chat input form ---
    with st.form(key="chat_form"):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_area("Enter your question:", height=70, key="input", label_visibility="collapsed")
        with col2:
            send = st.form_submit_button("Send")

        if send and user_input.strip():
            st.session_state['processing'] = True
            with st.spinner("Thinking..."):
                context = retrieve_context(user_input, index, chunks)
                st.session_state.conversation_history, last_response = chat_with_bot(user_input, st.session_state.conversation_history, context)
                st.session_state.suggested_questions = extract_suggested_questions(last_response)
            st.session_state['processing'] = False
            st.rerun()

if __name__ == "__main__":
    main()
