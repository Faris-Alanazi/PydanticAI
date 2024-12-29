import streamlit as st
import toml
import os
import asyncio
from agents import rag_agent

# Load configuration and set environment variable
config = toml.load("config.toml")
os.environ["OPENAI_API_KEY"] = config["API_KEYS"]["OPENAI"]

# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for user message
if prompt := st.chat_input("You:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = asyncio.run(rag_agent.run(prompt))
        message_placeholder.markdown(response.data)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.data})
