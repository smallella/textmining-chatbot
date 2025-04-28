# 💬 Chatbot template

A simple Streamlit app that shows how to build a chatbot using OpenAI's GPT-3.5.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

    - Note (commands for window users): if numpy installed with error:
      - Use python version == 3.10 and open venv to run 3.10 version
      ```
      $ py -3.10 -m venv .venv
      ```
      -  Run powershell as administrator and get authorized for venv-activation
      ```
      # Run in powershell as administrator
      $ Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
      ```

      ```
      # Run in terminal
      $ .venv\Scripts\activate
      ```

      - Deactivate to quit
      ```
      $ deactivate
      ```


2. Run the app (in venv)

   ```
   $ streamlit run streamlit_app.py
   ```
   - Press ctrl+C to stop the app

See deployed: https://ella-textmining-chatbot.streamlit.app/