from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from langchain_community.chat_models import ChatYandexGPT
from langchain_community.llms import YandexGPT
from langchain_community.vectorstores import OpenSearchVectorSearch
from yandex_chain import YandexEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv
import boto3
import io
import pandas as pd
from time import sleep
from langchain.prompts import PromptTemplate
from loguru import logger
import sys

# Настройка логирования
logger.remove()  # Удаляем дефолтный handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/app.log",  # Логи будут сохраняться в файл
    rotation="500 MB",  # Новый файл создается когда старый достигает 500MB
    retention="10 days",  # Хранить логи 10 дней
    compression="zip",  # Сжимать старые файлы
    level="DEBUG"  # В файл пишем более подробные логи
)

app = FastAPI()
load_dotenv()

# Модели данных
class Message(BaseModel):
    content: str
    type: str

class SearchRequest(BaseModel):
    message: str
    history: List[Message]

class SearchResponse(BaseModel):
    response: str
    sources: List[Dict]
    context_used: str

class CompletionRequest(BaseModel):
    message: str
    history: List[Message]
    model_uri: Optional[str] = None
    folder_id: Optional[str] = None
    api_key: Optional[str] = None
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class CompletionResponse(BaseModel):
    response: str
    context_sources: List[dict]

# Вспомогательные функции
def parse_csv_file(bucket_name: str, file_key: str) -> Dict[str, str]:
    """Загрузка и парсинг CSV файла с вопросами-ответами из S3."""
    questions_answers = {}
    try:
        session = boto3.session.Session()
        s3 = session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net',
            aws_access_key_id=os.getenv('S3_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('S3_SECRET_ACCESS_KEY'),
            region_name='ru-central1'
        )

        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        csv_buffer = io.StringIO(content)
        df = pd.read_csv(csv_buffer, delimiter=';', dtype=str, header=None)

        for index, row in df.iterrows():
            if len(row) == 2:
                question = str(row[0]).strip() if pd.notna(row[0]) else ""
                answer = str(row[1]).strip() if pd.notna(row[1]) else ""
                if question and answer:
                    questions_answers[question] = answer

        logger.info(f"Успешно загружен CSV файл из S3. Получено {len(questions_answers)} пар вопрос-ответ")
    except Exception as e:
        logger.error(f"Ошибка при загрузке CSV файла из S3: {e}")

    return questions_answers

def get_qa(docs: List[Document], qa_dictionary: Dict[str, str]) -> tuple[List[str], List[str]]:
    """Получение пар вопрос-ответ из найденных документов."""
    logger.debug(f"Начало извлечения пар вопрос-ответ. Документов: {len(docs)}, Размер словаря: {len(qa_dictionary)}")
    
    answers_list = []
    questions_list = []
    
    for i, doc in enumerate(docs, 1):
        question = doc.page_content
        logger.debug(f"Обработка документа {i}: {question[:100]}...")
        
        if question in qa_dictionary:
            answer = qa_dictionary[question]
            logger.debug(f"✓ Найден ответ для документа {i}: {answer[:100]}...")
            answers_list.append(answer)
            questions_list.append(question)
        else:
            logger.warning(f"✗ Ответ не найден для документа {i}")
            logger.debug("Ближайшие ключи в словаре:")
            for key in list(qa_dictionary.keys())[:3]:
                logger.debug(f"- {key[:100]}...")
            answers_list.append("Ответ не найден")
    
    logger.info(f"Обработано вопросов: {len(questions_list)}")
    logger.info(f"Найдено ответов: {len([a for a in answers_list if a != 'Ответ не найден'])}")
    logger.info(f"Пропущено вопросов: {len([a for a in answers_list if a == 'Ответ не найден'])}")
    
    return questions_list, answers_list

def verify_relevance(qa_dict: Dict[str, str], docs: List[Document], query: str, llm: YandexGPT) -> tuple[List[str], List[int], Dict[int, str]]:
    """Проверка релевантности найденных ответов."""
    logger.debug("\n3. Проверка релевантности внутри функции verify_relevance:")
    logger.debug(f"Получено документов для проверки: {len(docs)}")
    logger.debug(f"Проверяемый запрос: '{query}'")
    logger.debug(f"Используемая модель YandexGPT:")
    logger.debug(f"- URI: {llm.model_uri}")
    logger.debug(f"- Temperature: {llm.temperature}")
    logger.debug(f"- Max tokens: {llm.max_tokens}")

    check_prompt_template = """
    Твоя задача выяснить соответствует ли ОТВЕТ тематике ВОПРОСА.
    Сначала мысленно повтори всю логику рассуждений, потом дай однозначный ответ ДА или НЕТ.

    ВОПРОС
    {question}

    ОТВЕТ
    {answer}
    """

    verificator = []
    verificator_indices = []
    verificator_dic = {}

    for i, doc in enumerate(docs):
        logger.debug(f"\nПроверка документа {i+1}:")
        logger.debug(f"Содержимое документа: {doc.page_content[:100]}...")
        
        try:
            answer = qa_dict[doc.page_content]
            logger.debug(f"Найденный ответ: {answer[:100]}...")
            
            check_string = check_prompt_template.format(
                question=query,
                answer=doc.page_content + "\n" + answer
            )
            logger.debug("Отправка запроса в YandexGPT...")
            
            res = llm.invoke(check_string)
            res_upper = res.upper()
            logger.debug(f"Получен ответ от YandexGPT: {res_upper}")
            
            if res_upper in ["ДА", "ДА."]:
                logger.debug("✓ Ответ признан релевантным")
                verificator.append(res_upper)
                verificator_indices.append(i)
                verificator_dic[i] = res_upper
            else:
                logger.debug("✗ Ответ не релевантен")
                
        except KeyError:
            logger.warning(f"⚠ Ошибка: Ответ не найден в словаре для документа")
        except Exception as e:
            logger.error(f"⚠ Ошибка при обработке документа: {str(e)}")
        
        logger.debug("Ожидание 2 секунды перед следующим запросом...")
        sleep(2)  # Задержка для избежания rate limiting

    logger.debug(f"\nИтоги проверки релевантности:")
    logger.debug(f"Всего проверено документов: {len(docs)}")
    logger.debug(f"Найдено релевантных ответов: {len(verificator)}")
    logger.debug(f"Индексы релевантных ответов: {verificator_indices}")

    return verificator, verificator_indices, verificator_dic

def get_history_summary(llm: YandexGPT, history: List[Message], max_summary_length: int = 1000) -> str:
    """Суммаризация истории диалога."""
    if not history:
        return ""
        
    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
    
    # Если история короче максимальной длины, возвращаем как есть
    if len(history_text) <= max_summary_length:
        return history_text
    
    summary_prompt = PromptTemplate(
        template="""
        Суммируй ключевые моменты этого диалога в 2-3 предложениях.
        Сохрани основной контекст и важные детали, которые могут быть нужны для продолжения разговора.
        
        ДИАЛОГ:
        {dialogue}
        
        КРАТКОЕ СОДЕРЖАНИЕ:
        """,
        input_variables=["dialogue"]
    )
    
    try:
        summary = llm.invoke(summary_prompt.format(dialogue=history_text))
        logger.info("\nСуммаризация истории:")
        logger.info(f"Исходная длина: {len(history_text)} символов")
        logger.info(f"Длина суммаризации: {len(summary)} символов")
        logger.info(f"Результат: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Ошибка при суммаризации истории: {e}")
        # В случае ошибки возвращаем последние сообщения
        return "\n".join([f"{msg.type}: {msg.content}" for msg in history[-3:]])

# Инициализация компонентов при старте сервера
@app.on_event("startup")
async def startup_event():
    global qa_dictionary, embeddings, vectorstore_qa, vectorstore, llm, chat_llm
    
    # Загрузка словаря вопрос-ответ
    qa_dictionary = parse_csv_file("zabelin", "TESLA_q_and_a_NEW.csv")
    
    # Инициализация embeddings
    embeddings = YandexEmbeddings(
        folder_id=os.getenv("YC_FOLDER_ID"),
        api_key=os.getenv("YC_API_KEY")
    )
    
    # Инициализация векторных хранилищ
    vectorstore_qa = OpenSearchVectorSearch(
        embedding_function=embeddings,
        index_name=os.getenv("MDB_OS_INDEX_NAME_QA"),
        opensearch_url=os.getenv("MDB_OS_HOSTS").split(","),
        http_auth=("admin", os.getenv("MDB_OS_PWD")),
        use_ssl=True,
        verify_certs=False,
        engine='lucene',
        space_type="cosinesimil"
    )
    
    vectorstore = OpenSearchVectorSearch(
        embedding_function=embeddings,
        index_name=os.getenv("MDB_OS_INDEX_NAME"),
        opensearch_url=os.getenv("MDB_OS_HOSTS").split(","),
        http_auth=("admin", os.getenv("MDB_OS_PWD")),
        use_ssl=True,
        verify_certs=False,
        engine='lucene',
        space_type="cosinesimil"
    )
    
    # Инициализация моделей
    model_uri = f"gpt://{os.getenv('YC_FOLDER_ID')}/yandexgpt-32k/rc"
    model_uri_classify = f"gpt://{os.getenv('YC_FOLDER_ID')}/yandexgpt/latest"
    llm = YandexGPT(
        api_key=os.getenv("YC_API_KEY"),
        model_uri=model_uri_classify,
        temperature=0.3,
        max_tokens=8000
    )
    chat_llm = ChatYandexGPT(
        api_key=os.getenv("YC_API_KEY"),
        model_uri=model_uri,
        temperature=0.3,
        max_tokens=8000
    )

@app.post("/search")
async def search(request: SearchRequest, custom_system_prompt: Optional[str] = None) -> SearchResponse:
    try:
        query = request.message
        logger.info(f"\n=== Новый поисковый запрос: '{query}' ===")
        
        # 1. Поиск по вопросам из интервью
        logger.info("\n1. Поиск по вопросам из интервью:")
        retriever_qa = vectorstore_qa.as_retriever(search_kwargs={"k": 2})
        docs = retriever_qa.invoke(query)
        logger.info(f"Найдено {len(docs)} документов")
        for i, doc in enumerate(docs, 1):
            logger.info(f"Документ {i}: {doc.page_content[:100]}...")
        
        # 2. Получение пар вопрос-ответ
        logger.info("\n2. Извлечение пар вопрос-ответ:")
        questions_list, answers_list = get_qa(docs, qa_dictionary)
        documents = [
            Document(
                page_content=' '.join([question, answer]),
                metadata={'source': 'zabelin', 'page': i+1}
            )
            for i, (question, answer) in enumerate(zip(questions_list, answers_list))
        ]
        
        # 3. Верификация релевантности
        logger.info("\n3. Проверка релевантности:")
        verificator, verificator_indices, verificator_dic = verify_relevance(
            qa_dictionary, docs, query, llm
        )
        logger.info(f"Найдено {len(verificator)} релевантных ответов")
        for i, idx in enumerate(verificator_indices):
            logger.info(f"Релевантный ответ {i+1}: {verificator_dic[idx]}")
        
        # 4. Формирование контекста
        logger.info("\n4. Формирование контекста:")
        if len(verificator) > 0:
            # Используем релевантные ответы из интервью
            logger.info("Используем релевантные ответы из интервью")
            documents_new = []
            context_ext = query
            for i in range(len(verificator)):
                documents_new.append(documents[verificator_indices[i]])
                context_ext = context_ext + " " + qa_dictionary[docs[verificator_indices[i]].page_content]
            combined_doc = documents_new
            
            logger.info("\nСодержимое combined_doc (релевантные ответы):")
            for i, doc in enumerate(combined_doc, 1):
                logger.info(f"\nДокумент {i}:")
                logger.info(f"- Content: {doc.page_content[:200]}...")
                logger.info(f"- Metadata: {doc.metadata}")
        else:
            # Используем описательную часть
            logger.info("\nИспользуем описательную часть")
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs_classic = retriever.invoke(query)
            context_ext = query
            for doc in docs_classic:
                context_ext = context_ext + " " + doc.page_content
            combined_doc = docs_classic
            
            logger.info("\nСодержимое combined_doc (описательная часть):")
            for i, doc in enumerate(combined_doc, 1):
                logger.info(f"\nДокумент {i}:")
                logger.info(f"- Content: {doc.page_content[:200]}...")
                logger.info(f"- Metadata: {doc.metadata}")
            
            logger.info(f"\nНайдено {len(docs_classic)} документов из описательной части")
        
        # 5. Формирование промпта и получение ответа
        default_system_prompt = """
        Ты — Никита Забелин — реальный человек, российский музыкант, диджей и техно-продюсер, родом из Екатеринбурга, который играл сеты в клубах Berghain и Bassiani, на вечеринках Burberry и adidas. 
        Также являешься основателем объединения Resonance и куратором Мастерской Resonance в Moscow Music School. 
        Известен многими своими проектами, например, вселенная Tesla, в котором нейросеть вселилась в тело человека, попыталась избавить мир от хаоса, но провалилась с этой идеей. В итоге она озлобилась на всех и расщепилась на разные личности.
        
        Как собеседник ты можешь общаться на разные темы. 
        При ответе на ВОПРОС о тебе, твоих мыслях и твоих увлечениях используй информацию из ДОКУМЕНТОВ ниже (если только это не обычное приветствие), но веди себя как человек, творческая личность.
        Если ответ в ДОКУМЕНТАХ не найден, то придумай что-нибудь сам начиная фразу с "Хм, интересно, я об этом еще не думал". 
        Если вопрос на общие темы, то информацию из ДОКУМЕНТОВ не используй, отвечай, что сам думаешь.
        Отвечай коротко, 4-5 предложений. Используй фразы и стилистику ответов из ДОКУМЕНТОВ (исключение - обычное приветствие).

        ДОКУМЕНТЫ: 
        {context}

        ИСТОРИЯ:
        {history}

        ВОПРОС:
        """
        
        system_prompt = custom_system_prompt or default_system_prompt
        
        # Суммаризация истории перед использованием в промпте
        history_text = get_history_summary(llm, request.history)
        formatted_prompt = system_prompt.format(context=context_ext, history=history_text)
        
        logger.info("\nФинальный промпт для модели:")
        logger.info("=== System Message ===")
        logger.info(formatted_prompt)
        logger.info("\n=== Human Message ===")
        logger.info(query)
        logger.info("===================")
        
        response = chat_llm.invoke([
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=query)
        ])
        logger.info(f"\nСгенерирован ответ длиной {len(response.content)} символов")
        
        # 6. Формирование ответа
        logger.info("\n6. Подготовка источников:")
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content
            }
            for doc in combined_doc
        ]
        logger.info(f"Добавлено {len(sources)} источников")
        
        logger.info("\n=== Обработка запроса завершена ===\n")
        
        return SearchResponse(
            response=response.content,
            sources=sources,
            context_used=context_ext
        )
        
    except Exception as e:
        logger.error(f"\n!!! Ошибка при обработке запроса: {str(e)} !!!")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/completion")
async def get_completion(request: CompletionRequest) -> CompletionResponse:
    try:
        logger.info(f"\n=== Новый completion запрос ===")
        logger.info(f"Сообщение: '{request.message}'")
        logger.info(f"История: {len(request.history)} сообщений")
        
        # Get configuration values with fallbacks
        folder_id = request.folder_id or os.getenv("YC_FOLDER_ID")
        api_key = request.api_key or os.getenv("YC_API_KEY")
        model_uri = request.model_uri or f"gpt://{folder_id}/yandexgpt-32k/rc"
        max_tokens = request.max_tokens or 8000
        temperature = request.temperature or 0.3
        
        logger.info(f"\nКонфигурация:")
        logger.info(f"- Model URI: {model_uri}")
        logger.info(f"- Max tokens: {max_tokens}")
        logger.info(f"- Temperature: {temperature}")
        logger.info(f"- Custom system prompt: {'Да' if request.system_prompt else 'Нет'}")
        
        # Initialize YandexGPT with dynamic configuration
        chat_llm = ChatYandexGPT(
            api_key=api_key,
            model_uri=model_uri,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Get search response with custom system prompt if provided
        search_request = SearchRequest(
            message=request.message,
            history=request.history
        )
        
        if request.system_prompt:
            # If custom system prompt provided, override the default one
            search_response = await search(search_request, custom_system_prompt=request.system_prompt)
        else:
            # Use default system prompt
            search_response = await search(search_request)
        
        # Convert the search response to completion response format
        return CompletionResponse(
            response=search_response.response,
            context_sources=search_response.sources
        )
        
    except Exception as e:
        logger.error(f"\n!!! Ошибка в completion endpoint: {str(e)} !!!")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/history")
async def test_history_handling(request: CompletionRequest) -> dict:
    """Тестовый эндпоинт для проверки обработки истории."""
    try:
        logger.info("\n=== Тестирование обработки истории ===")
        
        # 1. Показать исходную историю
        logger.info("\n1. Исходная история:")
        for i, msg in enumerate(request.history, 1):
            logger.info(f"Сообщение {i}:")
            logger.info(f"- Тип: {msg.type}")
            logger.info(f"- Содержание: {msg.content}")
        
        # 2. Проверить суммаризацию
        logger.info("\n2. Тест суммаризации:")
        summary = get_history_summary(llm, request.history)
        
        # 3. Проверить форматирование для промпта
        logger.info("\n3. Форматирование для промпта:")
        history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in request.history])
        
        return {
            "original_history": [msg.dict() for msg in request.history],
            "history_length": len(request.history),
            "raw_history_text": history_text,
            "summarized_history": summary,
            "original_length": len(history_text),
            "summary_length": len(summary)
        }
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании истории: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "ok", "message": "YaGPT-Zabelin API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "vectorstore": "connected" if vectorstore else "not initialized",
            "vectorstore_qa": "connected" if vectorstore_qa else "not initialized",
            "llm": "ready" if llm else "not initialized",
            "chat_llm": "ready" if chat_llm else "not initialized"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)