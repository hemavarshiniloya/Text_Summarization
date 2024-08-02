import streamlit as st
from txtai.pipeline import Summary
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import xml.etree.ElementTree as ET
import os
import pytesseract
from googletrans import Translator
import re
import torch
from PIL import Image

# List of languages with their ISO 639-1 codes
languages = {
    "English": "en", 
    "Afrikaans": "af",
    # (Other languages)
    "Zulu": "zu"
}

# Set page configuration
st.set_page_config(layout="wide")

# Initialize text summarizer
def text_summary(text, maxlength=None):
    summary = Summary()
    result = summary(text)
    return result

# Initialize tokenizer and model for sentiment analysis
def initialize_sentiment_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
        return tokenizer, model
    except Exception as e:
        st.error(f"An error occurred while loading the sentiment model: {str(e)}")
        return None, None

# Perform sentiment analysis
def sentiment_analysis(text, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        return {"label": "POSITIVE" if predictions.item() == 1 else "NEGATIVE", "score": torch.softmax(logits, dim=1)[0, predictions.item()].item()}
    except Exception as e:
        st.error(f"An error occurred during sentiment analysis: {str(e)}")
        return {"label": "ERROR", "score": 0.0}

# Function to preprocess text
def preprocess_text(text):
    # Remove extra whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        text = "\n".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to extract text from TXT
def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Function to extract text from HTML
def extract_text_from_html(file):
    soup = BeautifulSoup(file.read(), "html.parser")
    paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    text = "\n".join([p.get_text() for p in paragraphs])
    return text

# Function to extract text from CSV
def extract_text_from_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

# Function to extract text from XML
def extract_text_from_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    text = " ".join([elem.text for elem in root.iter() if elem.text])
    return text

# Function to extract text from Image
def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

# Function to save summary to history
def save_summary(summary):
    filename = "summary_history.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(summary + "\n\n")

# Function to load summary history
def load_summary_history():
    filename = "summary_history.txt"
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback if UTF-8 decoding fails
            with open(filename, "r", encoding="latin1") as f:
                return f.read()
    return ""

# Function to clear summary history
def clear_summary_history():
    filename = "summary_history.txt"
    if os.path.exists(filename):
        os.remove(filename)

# Function to clear input fields based on choice
def clear_input(choice):
    if choice == "Summarize Text":
        st.session_state.text = ""
    elif choice == "Summarize URL":
        st.session_state.url = ""
    elif choice == "Summarize Document":
        st.session_state.uploaded_files = []
    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = ""

# Function to validate input
def validate_input(text):
    return bool(text and text.strip())

# Function to translate text using Google Translate API
def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Function to download file
def download_file(content, filename):
    st.download_button(label="Download Summary", data=content, file_name=filename, mime="text/plain")

# Main function to handle Streamlit app
def main():
    st.title("Text Summarization and Sentiment Analysis App")
    st.sidebar.title("Options")

    # Initialize session state variables if not already set
    if 'text' not in st.session_state:
        st.session_state.text = ""
    if 'url' not in st.session_state:
        st.session_state.url = ""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'clipboard_text' not in st.session_state:
        st.session_state.clipboard_text = ""
    
    # Handle choice selection
    choice = st.sidebar.radio("Choose an option", ["Summarize Text", "Summarize URL", "Summarize Document", "Summarize Text from Clipboard"])

    # Load sentiment model
    tokenizer, model = initialize_sentiment_model()

    # Display input fields and options based on choice
    if choice == "Summarize Text":
        st.session_state.text = st.text_area("Enter Text", st.session_state.text)
        st.session_state.language = st.selectbox("Select Language for Summary", list(languages.keys()))
        
        if st.button("Summarize Text"):
            if validate_input(st.session_state.text):
                with st.spinner("Summarizing text..."):
                    text = preprocess_text(st.session_state.text)
                    summary = text_summary(text)
                    
                    # Display summary
                    st.write("### Summary")
                    st.write(summary)

                    # Sentiment Analysis
                    if tokenizer and model:
                        sentiment = sentiment_analysis(text, tokenizer, model)
                        st.write("### Sentiment Analysis")
                        st.write(f"Label: {sentiment['label']}, Score: {sentiment['score']:.2f}")

                    # Translate summary
                    target_language = languages.get(st.session_state.language, "en")
                    translated_summary = translate_text(summary, target_language)
                    st.write("### Translated Summary")
                    st.write(translated_summary)

                    save_summary(summary)
                    download_file(summary, "summary.txt")

    elif choice == "Summarize URL":
        st.session_state.url = st.text_input("Enter URL", st.session_state.url)
        st.session_state.language = st.selectbox("Select Language for Summary", list(languages.keys()))
        
        if st.button("Summarize URL"):
            if validate_input(st.session_state.url):
                with st.spinner("Extracting text from URL..."):
                    text = extract_text_from_url(st.session_state.url)
                    if text:
                        with st.spinner("Summarizing text..."):
                            text = preprocess_text(text)
                            summary = text_summary(text)
                            
                            # Display summary
                            st.write("### Summary")
                            st.write(summary)

                            # Sentiment Analysis
                            if tokenizer and model:
                                sentiment = sentiment_analysis(text, tokenizer, model)
                                st.write("### Sentiment Analysis")
                                st.write(f"Label: {sentiment['label']}, Score: {sentiment['score']:.2f}")

                            # Translate summary
                            target_language = languages.get(st.session_state.language, "en")
                            translated_summary = translate_text(summary, target_language)
                            st.write("### Translated Summary")
                            st.write(translated_summary)

                            save_summary(summary)
                            download_file(summary, "summary.txt")
                    else:
                        st.error("Failed to extract text from the URL.")

    elif choice == "Summarize Document":
        st.session_state.uploaded_files = st.file_uploader("Upload Document", type=["pdf", "docx", "txt", "html", "csv", "xml", "png", "jpg"], accept_multiple_files=True)
        st.session_state.language = st.selectbox("Select Language for Summary", list(languages.keys()))
        
        if st.button("Summarize Document"):
            if st.session_state.uploaded_files:
                with st.spinner("Extracting text from document..."):
                    text = ""
                    for file in st.session_state.uploaded_files:
                        if file.type == "application/pdf":
                            text += extract_text_from_pdf(file)
                        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            text += extract_text_from_docx(file)
                        elif file.type == "text/plain":
                            text += extract_text_from_txt(file)
                        elif file.type == "text/html":
                            text += extract_text_from_html(file)
                        elif file.type == "text/csv":
                            text += extract_text_from_csv(file)
                        elif file.type == "application/xml":
                            text += extract_text_from_xml(file)
                        elif file.type in ["image/png", "image/jpeg"]:
                            text += extract_text_from_image(file)
                    
                    if text:
                        with st.spinner("Summarizing text..."):
                            text = preprocess_text(text)
                            summary = text_summary(text)
                            
                            # Display summary
                            st.write("### Summary")
                            st.write(summary)

                            # Sentiment Analysis
                            if tokenizer and model:
                                sentiment = sentiment_analysis(text, tokenizer, model)
                                st.write("### Sentiment Analysis")
                                st.write(f"Label: {sentiment['label']}, Score: {sentiment['score']:.2f}")

                            # Translate summary
                            target_language = languages.get(st.session_state.language, "en")
                            translated_summary = translate_text(summary, target_language)
                            st.write("### Translated Summary")
                            st.write(translated_summary)

                            save_summary(summary)
                            download_file(summary, "summary.txt")
                    else:
                        st.error("Failed to extract text from the document.")

    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = st.text_area("Paste Clipboard Text Here", st.session_state.clipboard_text)
        st.session_state.language = st.selectbox("Select Language for Summary", list(languages.keys()))
        
        if st.button("Summarize Text from Clipboard"):
            if validate_input(st.session_state.clipboard_text):
                with st.spinner("Summarizing text..."):
                    text = preprocess_text(st.session_state.clipboard_text)
                    summary = text_summary(text)
                    
                    # Display summary
                    st.write("### Summary")
                    st.write(summary)

                    # Sentiment Analysis
                    if tokenizer and model:
                        sentiment = sentiment_analysis(text, tokenizer, model)
                        st.write("### Sentiment Analysis")
                        st.write(f"Label: {sentiment['label']}, Score: {sentiment['score']:.2f}")

                    # Translate summary
                    target_language = languages.get(st.session_state.language, "en")
                    translated_summary = translate_text(summary, target_language)
                    st.write("### Translated Summary")
                    st.write(translated_summary)

                    save_summary(summary)
                    download_file(summary, "summary.txt")

    # Clear Input button
    if st.sidebar.button("Clear Input"):
        clear_input(choice)
    
    # Clear Summary History button
    if st.sidebar.button("Clear Summary History"):
        clear_summary_history()

    # Load Summary History button
    if st.sidebar.button("Load Summary History"):
        history = load_summary_history()
        if history:
            st.write("### Summary History")
            st.write(history)
        else:
            st.write("No history available.")

if __name__ == "__main__":
    main()
