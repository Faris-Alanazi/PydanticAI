import streamlit as st
import toml
import os
import asyncio
from agents import rag_agent

# Load configuration and set environment variable
config = toml.load("config.toml")
os.environ["OPENAI_API_KEY"] = config["API_KEYS"]["OPENAI"]

# Chat Input for user message
if prompt := st.chat_input("You:"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = asyncio.run(rag_agent.run_with_history(prompt))
        message_placeholder.markdown(response)

    # Debugging: Optional - Display conversation history in the app
    st.json(rag_agent.get_history())
