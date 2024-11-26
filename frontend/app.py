import streamlit as st
import requests
from typing import List
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from PIL import Image

@dataclass
class Message:
    content: str
    type: str

def get_completion(message: str, history: List[Message]) -> dict:
    """Send request and get completion."""
    # Ограничиваем историю последними N сообщениями
    MAX_HISTORY_LENGTH = 10
    filtered_history = history[-MAX_HISTORY_LENGTH:] if len(history) > MAX_HISTORY_LENGTH else history
    
    print("\nОтправка запроса в бэкенд:")
    print(f"Сообщение: {message}")
    print(f"История (последние {len(filtered_history)} сообщений):")
    for msg in filtered_history:
        print(f"- {msg.type}: {msg.content[:100]}...")
    
    response = requests.post(
        "http://localhost:8000/completion",
        json={
            "message": message,
            "history": [{"content": msg.content, "type": msg.type} for msg in filtered_history]
        }
    )
    return response.json()

def get_voice(text: str, voice: str = "zabelin") -> bytes:
    """Get voice synthesis from TTS service."""
    response = requests.get(
        os.getenv("SK_API_EP"),
        params={"text": text, "voice": voice}
    )
    return response.content

def main():
    # Setup page
    logo = Image.open('./images/logo.png')
    resized_logo = logo.resize((100, 100))
    st.set_page_config(page_title="Забелин чат-бот", page_icon="")
    st.image(resized_logo)
    st.title('📖 Забелин чат-бот')
    
    # Initialize session state with additional parameters
    if "messages" not in st.session_state:
        st.session_state.messages = [Message(content="Привет, я Никита Забелин! Чем могу вам помочь?", type="ai")]
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
    
    # Add conversation management UI
    with st.sidebar:
        if st.button("Начать новый диалог"):
            st.session_state.messages = [Message(content="Привет, я Никита Забелин! Чем могу вам помочь?", type="ai")]
            st.session_state.conversation_started = False
            st.rerun()
    
    # Display messages
    for msg in st.session_state.messages:
        st.chat_message(msg.type).write(msg.content)
    
    # Handle user input
    if prompt := st.chat_input():
        # Add user message
        st.chat_message("human").write(prompt)
        st.session_state.messages.append(Message(content=prompt, type="human"))
        
        # Get completion from backend
        response = get_completion(prompt, st.session_state.messages)
        
        # Display AI response
        st.chat_message("ai").write(response["response"])
        st.session_state.messages.append(Message(content=response["response"], type="ai"))
        
        # Handle voice synthesis
        reply_audio = get_voice(response["response"])
        question_audio = get_voice(prompt)
        
        # Create audio players
        col1, col2 = st.columns(2)
        with col1:
            st.write("Озвучить ответ:")
            st.audio(reply_audio, format="audio/mpeg", loop=False)
        with col2:
            st.write("Озвучить вопрос:")
            st.audio(question_audio, format="audio/mpeg", loop=False)
        
        # Display sources
        for i, source in enumerate(response["context_sources"], 1):
            with st.expander(f"**Источник N{i}:** [{source['source']}]"):
                st.write(source["content"])

if __name__ == "__main__":
    try:
        load_dotenv()
        main()
    except Exception as e:
        st.error(f"Произошла ошибка: {str(e)}")