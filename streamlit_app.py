import streamlit as st
from openai import OpenAI
import time
import re
import requests
import fitz
import json
from db_utils import init_db, get_user_profile, save_user_profile
from qa_utils.Word2vec import view_2d, view_3d, cbow, skipgram, negative_sampling
from ui_utils import render_pdf_upload_section, show_dismissible_alert
from pdf_context import *
from response_generator import generate_response

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

def is_valid_image_url(url):
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200 and 'image' in response.headers["Content-Type"]:
            return True
        else:
            return False
    except:
        return False

def load_example_from_json(json_path, key):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(key, "")

def render_sidebar():
    with st.sidebar:
        st_c_1 = st.container(border=True)
        with st_c_1:
            user_image = st.session_state.get("user_image", "https://www.w3schools.com/howto/img_avatar.png")
            if user_image and is_valid_image_url(user_image):
                st.image(user_image)
            else:
                show_dismissible_alert(
                    "avatar_warning",
                    "⚠️ Invalid avatar URL.<br>Showing default image.<br>Image Ref: <a href='https://unsplash.com/' target='_blank'>https://unsplash.com/</a>",
                    alert_type="warning"
                )
                st.image("https://www.w3schools.com/howto/img_avatar.png")

        st.markdown("---")

        with st.expander("📦 Vector Semantics - Word2vec", expanded=False):
            if st.button("🧭 Vector space - 2D View"):
                clear_vector_session_state()
                st.session_state["vector_task_function"] = view_2d.run
            if st.button("🧭 Vector space - 3D View"):
                clear_vector_session_state()
                st.session_state["vector_task_function"] = view_3d.run
            if st.button("📘 CBOW"):
                clear_vector_session_state()
                st.session_state["vector_task_function"] = cbow.run
            if st.button("⚙️ Skipgram"):
                clear_vector_session_state()
                st.session_state["vector_task_function"] = skipgram.run
            if st.button("🛠️ Negative Sampling"):
                clear_vector_session_state()
                st.session_state["vector_task_function"] = negative_sampling.run


        st.markdown("---")
        selected_lang = st.selectbox("🌐 Language", ["English", "繁體中文"], index=1)
        st.session_state['lang_setting'] = selected_lang

        with st.expander("🧑‍💻 Profile Settings", expanded=False):
            with st.form(key="profile_form"):
                new_name = st.text_input("User Name", value=st.session_state.get("user_name", "Ella"))
                new_image = st.text_input("Avatar Image URL", value=st.session_state.get("user_image", ""))
                submitted = st.form_submit_button("💾 Save Profile")

                if submitted:
                    save_user_profile(new_name, new_image)
                    st.session_state["user_name"] = new_name
                    st.session_state["user_image"] = new_image
                    st.success("Profile saved! Please refresh to see changes.")
                    st.rerun()

def render_vector_task_section():
    if "vector_task_function" not in st.session_state:
        return

    st.markdown("## 🧠 Provide your own sentences for Word2Vec")

    # 初始化 input 區塊
    st.session_state.setdefault("user_input_text", "")
    st.session_state.setdefault("input_sentences", [])

    if st.button("🔖 Load Example Sentences"):
        example_text = load_example_from_json("db/examples.json", "vector semantic example")
        st.session_state["user_input_text"] = example_text

    user_input_text = st.text_area(
        label="Enter sentences (one per line):",
        value=st.session_state["user_input_text"],
        height=300,
        placeholder="Type one sentence per line..."
    )
    st.session_state["user_input_text"] = user_input_text

    if st.button("🚀 Run Vector Task"):
        if user_input_text.strip():
            input_sentences = [line.strip() for line in user_input_text.splitlines() if line.strip()]
            st.session_state["input_sentences"] = input_sentences
        else:
            st.warning("⚠️ Please enter some sentences before running the vector task.")

    if st.session_state.get("input_sentences"):
        st.session_state["vector_task_function"](sentences=st.session_state["input_sentences"])

def render_chat_section():
    st_c_chat = st.container(border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st_c_chat.chat_message(msg["role"], avatar=st.session_state.get("user_image", "")).markdown((msg["content"]))
            elif msg["role"] == "assistant":
                st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            else:
                image_tmp = msg.get("image")
                if image_tmp:
                    st_c_chat.chat_message(msg["role"], avatar=image_tmp).markdown((msg["content"]))
                else:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))

    def chat(prompt: str):
        chat_user_image = st.session_state.get("user_image", "https://www.w3schools.com/howto/img_avatar.png")
        st_c_chat.chat_message("user", avatar=chat_user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = generate_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st_c_chat.chat_message("assistant").write_stream(stream_data(response))

    if prompt := st.chat_input(placeholder="Please input your command", key="chat_bot"):
        chat(prompt)

def clear_vector_session_state():
    """清除跟 Vector 任務有關的所有 session_state 變數"""
    keys_to_clear = [
        "input_sentences",
        "user_input_text",
        "selected_indices_3d",
        "sentence_picker",
        "trigger_plot_3d"
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

def main():
    st.set_page_config(
        page_title='K-Assistant - The Residemy Agent',
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Get Help': 'https://streamlit.io/',
            'Report a bug': 'https://github.com',
            'About': 'About your application: **Hello world**'
        },
        page_icon="img/favicon.ico"
    )
    init_db()
    profile = get_user_profile()

    if profile:
        st.session_state.setdefault("user_name", profile.get("user_name", "Ella"))
        st.session_state.setdefault("user_image", profile.get("user_image", "https://images.unsplash.com/photo-1602847213180-50e43a80bef4?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8bGFtYnxlbnwwfHwwfHx8MA%3D%3D"))
    else:
        st.session_state.setdefault("user_name", "Ella")
        st.session_state.setdefault("user_image", "https://images.unsplash.com/photo-1602847213180-50e43a80bef4?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8bGFtYnxlbnwwfHwwfHx8MA%3D%3D")

    st.title(f"💬 {st.session_state['user_name']}'s Chatbot")

    render_sidebar()
    render_pdf_upload_section()
    render_chat_section()
    render_vector_task_section()

    if "pending_vector_task" in st.session_state:
        st.session_state["vector_task_function"] = st.session_state["pending_vector_task"]
        del st.session_state["pending_vector_task"]
        st.rerun()

if __name__ == "__main__":
    main()
 