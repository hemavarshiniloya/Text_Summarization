FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader punkt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "your_app_file.py", "--server.port=8501", "--server.address=0.0.0.0"]
