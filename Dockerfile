FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a script to download NLTK and TextBlob data
RUN echo '\
import nltk\n\
import textblob\n\
nltk.download("punkt")\n\
nltk.download("averaged_perceptron_tagger")\n\
nltk.download("brown")\n\
nltk.download("wordnet")\n\
nltk.download("omw-1.4")\n\
import subprocess\n\
subprocess.run(["python", "-m", "textblob.download_corpora"])\
' > download_nltk_data.py

# Run the script to download data
RUN python download_nltk_data.py

# Copy the rest of the application
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
