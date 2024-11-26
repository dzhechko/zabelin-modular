# YandexGPT Zabelin Chat Bot 🎵🤖

Интерактивный чат-бот, имитирующий общение с Никитой Забелиным - известным российским музыкантом, диджеем и техно-продюсером. Бот использует технологии YandexGPT и векторные базы данных для генерации контекстно-релевантных ответов на основе реальных интервью и материалов о Никите.

## 🌟 Основные возможности

- 💬 Естественное общение в формате чата
- 🎯 Контекстно-релевантные ответы на основе реальных интервью
- 🔍 Умный поиск и верификация релевантности ответов
- 🗣️ Синтез речи для вопросов и ответов
- 📚 Использование истории диалога для поддержания контекста
- 🔄 Автоматическая суммаризация длинных диалогов

## 🏗️ Архитектура

Проект состоит из двух основных компонентов:

### Backend (FastAPI)
- Обработка запросов и генерация ответов
- Интеграция с YandexGPT
- Векторный поиск по базе знаний
- Верификация релевантности ответов
- Управление историей диалога

### Frontend (Streamlit)
- Пользовательский интерфейс в формате чата
- Отображение источников информации
- Воспроизведение синтезированной речи
- Управление сессией диалога

## 🚀 Запуск проекта

### Предварительные требования 
- Python 3.9+
- Docker и Docker Compose
- Учетные данные Yandex Cloud
- Доступ к S3-совместимому хранилищу

### Установка и настройка

Создание виртуального окружения
```
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
```
Установка зависимостей
`pip install -r requirements.txt`


### Настройка окружения

Создайте файл `.env` в корневой директории:
```
# Yandex Cloud
YC_API_KEY=your_api_key
YC_FOLDER_ID=your_folder_id

# OpenSearch
MDB_OS_HOSTS=your_opensearch_hosts
MDB_OS_PWD=your_opensearch_password
MDB_OS_INDEX_NAME=your_index_name
MDB_OS_INDEX_NAME_QA=your_qa_index_name

# S3 Storage
S3_ACCESS_KEY_ID=your_s3_key
S3_SECRET_ACCESS_KEY=your_s3_secret

# Speech Synthesis
SK_API_EP=your_speech_synthesis_endpoint
```

### Запуск приложения

1. Запуск бэкенда:
```
cd backend
uvicorn main:app --reload
```

2. Запуск фронтенда (в отдельном терминале):
```
cd frontend
streamlit run app.py
```

Приложение будет доступно по адресу: `http://localhost:8501`

## 🔍 API Endpoints

- `POST /completion` - Основной эндпоинт для генерации ответов
- `POST /search` - Поиск по базе знаний
- `POST /test/history` - Тестирование обработки истории
- `GET /health` - Проверка состояния сервиса
- `GET /` - Проверка работоспособности API

## 🛠️ Разработка и тестирование

### Тестирование API
Проверка здоровья сервиса
```
curl http://localhost:8000/health
```
Тестирование обработки истории
```
curl -X POST
"http://localhost:8000/test/history" \
-H "Content-Type: application/json" \
-d '{
"message": "Тестовое сообщение",
"history": [
{"type": "human", "content": "Привет!"},
{"type": "ai", "content": "Здравствуйте!"}
]
}'
```

### Мониторинг
- Все важные операции логируются в консоль
- Доступна подробная информация о каждом этапе обработки запроса
- Отслеживание состояния компонентов через `/health` endpoint

## 📚 Дополнительная информация

- Используется модель YandexGPT для генерации ответов
- Векторное хранилище на базе OpenSearch
- Хранение данных в Yandex Object Storage (S3)
- Асинхронная обработка запросов с помощью FastAPI
- Интерактивный UI на базе Streamlit

## 🤝 Вклад в проект

Мы открыты для улучшений! Создавайте issues и pull requests.

## 📝 Лицензия

MIT License