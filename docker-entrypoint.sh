#!/bin/bash
set -e

# Ожидание доступности backend (если запускается frontend)
if [[ "$*" == *"streamlit"* ]]; then
    echo "Waiting for backend..."
    until curl -f http://backend:8000/health; do
        sleep 1
    done
    echo "Backend is up!"
fi

# Выполнение команды
exec "$@" 