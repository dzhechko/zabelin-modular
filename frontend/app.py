import streamlit as st
import requests
from typing import List
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from PIL import Image
from loguru import logger
import sys

# Настройка логирования
logger.remove()  # Удаляем дефолтный handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>frontend:{function}:{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/frontend.log",
    rotation="500 MB",
    retention="10 days",
    compression="zip",
    level="DEBUG"
)

@dataclass
class Message:
    content: str
    type: str

def get_completion(message: str, history: List[Message]) -> dict:
    """Send request and get completion."""
    # Ограничиваем историю последними N сообщениями
    MAX_HISTORY_LENGTH = 10
    filtered_history = history[-MAX_HISTORY_LENGTH:] if len(history) > MAX_HISTORY_LENGTH else history
    
    logger.info("Отправка запроса в бэкенд")
    logger.info(f"Сообщение: {message}")
    logger.info(f"История (последние {len(filtered_history)} сообщений):")
    for msg in filtered_history:
        logger.debug(f"- {msg.type}: {msg.content[:100]}...")
    
    try:
        response = requests.post(
            "http://backend:8000/completion",
            json={
                "message": message,
                "history": [{"content": msg.content, "type": msg.type} for msg in filtered_history]
            }
        )
        response.raise_for_status()  # Проверяем статус ответа
        logger.info("Успешно получен ответ от бэкенда")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при запросе к бэкенду: {str(e)}")
        raise

def get_voice(text: str, voice: str = "zabelin") -> bytes:
    """Get voice synthesis from TTS service."""
    try:
        logger.info(f"Запрос синтеза речи для текста длиной {len(text)} символов")
        response = requests.get(
            os.getenv("SK_API_EP"),
            params={"text": text, "voice": voice}
        )
        response.raise_for_status()
        logger.info("Успешно получен аудио-ответ")
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при запросе синтеза речи: {str(e)}")
        raise

def main():
    try:
        # Setup page
        logger.info("Инициализация веб-интерфейса")
        logo = Image.open('frontend/images/logo.png')
        resized_logo = logo.resize((100, 100))
        st.set_page_config(page_title="Забелин чат-бот", page_icon="")
        st.image(resized_logo)
        st.title('📖 Забелин чат-бот')
        
        # Initialize session state with additional parameters
        if "messages" not in st.session_state:
            logger.info("Инициализация новой сессии")
            st.session_state.messages = [Message(content="Привет, я Никита Забелин! Чем могу вам помочь?", type="ai")]
        if "conversation_started" not in st.session_state:
            st.session_state.conversation_started = False
        
        # Add conversation management UI
        with st.sidebar:
            if st.button("Начать новый диалог"):
                logger.info("Пользователь начал новый диалог")
                st.session_state.messages = [Message(content="Привет, я Никита Забелин! Чем могу вам помочь?", type="ai")]
                st.session_state.conversation_started = False
                st.rerun()
        
        # Display messages
        for msg in st.session_state.messages:
            st.chat_message(msg.type).write(msg.content)
        
        # Handle user input
        if prompt := st.chat_input():
            logger.info(f"Получен новый запрос от пользователя: {prompt[:100]}...")
            
            # Add user message
            st.chat_message("human").write(prompt)
            st.session_state.messages.append(Message(content=prompt, type="human"))
            
            # Get completion from backend
            response = get_completion(prompt, st.session_state.messages)
            logger.debug(f"Получен ответ от бэкенда длиной {len(response['response'])} символов")
            
            # Display AI response
            st.chat_message("ai").write(response["response"])
            st.session_state.messages.append(Message(content=response["response"], type="ai"))
            
            # Handle voice synthesis
            logger.info("Запрос синтеза речи для ответа и вопроса")
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
            logger.debug(f"Отображение {len(response['context_sources'])} источников")
            for i, source in enumerate(response["context_sources"], 1):
                with st.expander(f"**Источник N{i}:** [{source['source']}]"):
                    st.write(source["content"])

    except Exception as e:
        logger.error(f"Критическая ошибка в main(): {str(e)}")
        st.error(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    try:
        load_dotenv()
        logger.info("Запуск приложения")
        main()
    except Exception as e:
        logger.error(f"Ошибка при запуске приложения: {str(e)}")
        st.error(f"Произошла ошибка: {str(e)}")