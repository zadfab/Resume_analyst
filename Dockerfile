FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y sqlite3 libsqlite3-dev

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir chromadb langchain

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8501

CMD ["streamlit","run","index.py","--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]