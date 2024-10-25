# Start with an official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# Set the working directory
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt /app/requirements.txt

# Install dependencies and download NLTK data
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m nltk.downloader punkt

# Copy the entire app code to the container
COPY . /app

# Expose the port on which Streamlit will run
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
