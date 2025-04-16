import streamlit as st
from openai import OpenAI
import time
import re
import requests
from db_utils import init_db, get_user_profile, save_user_profile
from ui_utils import show_dismissible_alert
from response_generator import pdf_upload_section, generate_response

placeholderstr = "Please input your command"
# user_name = "Brian"
# user_image = "https://www.w3schools.com/howto/img_avatar.png"

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

    # Get User Profile from db
    init_db()
    profile = get_user_profile()

    if "user_name" not in st.session_state:
        st.session_state["user_name"] = profile["user_name"] if profile else "Brian"
    if "user_image" not in st.session_state:
        st.session_state["user_image"] = profile["user_image"] if profile else "https://www.w3schools.com/howto/img_avatar.png"

    # Show title and description.
    user_name = st.session_state["user_name"]
    user_image = st.session_state["user_image"]
    st.title(f"💬 {user_name}'s Chatbot")

    with st.sidebar:
        st_c_1 = st.container(border=True)
        with st_c_1:
            if user_image:
                if is_valid_image_url(user_image):
                    st.image(user_image)
                else:
                    # st.warning("⚠️ Invalid avatar URL. Showing default image.")
                    # show_dismissible_alert("⚠️ Invalid avatar URL. Showing default image.<br>Image Ref: https://unsplash.com/", alert_type="warning")
                    show_dismissible_alert(
                        "avatar_warning",
                        "⚠️ Invalid avatar URL.<br>Showing default image.<br>Image Ref: <a href='https://unsplash.com/' target='_blank'>https://unsplash.com/</a>",
                        alert_type="warning"
                    )
                    st.image("https://www.w3schools.com/howto/img_avatar.png")
            else:
                st.image("https://www.w3schools.com/howto/img_avatar.png")

        st.markdown("---")
        # st.write("🌐 Language")
        selected_lang = st.selectbox("🌐 Language", ["English", "繁體中文"], index=1)
        st.session_state['lang_setting'] = selected_lang

        # st.header("🧑‍💻 Profile Settings")
        with st.expander("🧑‍💻 Profile Settings", expanded=False):
            with st.form(key="profile_form"):
                new_name = st.text_input("User Name", value=st.session_state["user_name"])
                new_image = st.text_input("Avatar Image URL", value=st.session_state["user_image"])
                submitted = st.form_submit_button("💾 Save Profile")

                if submitted:
                    save_user_profile(new_name, new_image)
                    st.session_state["user_name"] = new_name
                    st.session_state["user_image"] = new_image
                    st.success("Profile saved! Please refresh to see changes.")
                    st.rerun()

    st_c_chat = st.container(border=True)
    pdf_upload_section()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                if user_image:
                    st_c_chat.chat_message(msg["role"],avatar=user_image).markdown((msg["content"]))
                else:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            elif msg["role"] == "assistant":
                st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            else:
                try:
                    image_tmp = msg.get("image")
                    if image_tmp:
                        st_c_chat.chat_message(msg["role"],avatar=image_tmp).markdown((msg["content"]))
                except:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))

    # test generate_response function
    # def generate_response(prompt):
    #     pattern = r'\b(i(\'?m| am| feel| think i(\'?)?m)?\s*(so\s+)?(stupid|ugly|dumb|idiot|worthless|loser|useless))\b'
    #     if re.search(pattern, prompt, re.IGNORECASE):
    #         return "Yes, you are!"
    #     else:
    #         return f"You say: {prompt}."
    # response = generate_response(prompt)

    # Chat function section (timing included inside function)
    def chat(prompt: str):
        if user_image and is_valid_image_url(user_image):
            chat_user_image = user_image
        else:
            chat_user_image = "https://www.w3schools.com/howto/img_avatar.png"
        st_c_chat.chat_message("user", avatar=chat_user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call generate_response function
        response = generate_response(prompt)
        # response = f"You type: {prompt}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st_c_chat.chat_message("assistant").write_stream(stream_data(response))


    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)

if __name__ == "__main__":
    main()