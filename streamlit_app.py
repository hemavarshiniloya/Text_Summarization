import streamlit as st
from txtai.pipeline import Summary
from transformers import pipeline as hf_pipeline
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import xml.etree.ElementTree as ET
import os
import pytesseract
from PIL import Image
from googletrans import Translator, LANGUAGES
import re

# List of languages with their ISO 639-1 codes
languages = {name.capitalize(): code for name, code in LANGUAGES.items()}

# Initialize text summarizer
def text_summary(text, maxlength=None):
    summary = Summary()
    result = summary(text)
    return result

# Initialize sentiment analyzer
def sentiment_analysis(text):
    sentiment_pipeline = hf_pipeline("sentiment-analysis")
    try:
        # Limit input length to avoid issues with long texts
        if len(text) > 512:
            text = text[:512]
        result = sentiment_pipeline(text)
        return result
    except Exception as e:
        st.error(f"An error occurred during sentiment analysis: {str(e)}")
        return {"label": "Unknown", "score": 0}

# Function to preprocess text
def preprocess_text(text):
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
            with open(filename, "r", encoding="latin1") as f:
                return f.read()
    return ""

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
    try:
        translator = Translator()
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        st.error(f"An error occurred during translation: {str(e)}")
        return None

# Function to download file
def download_file(content, filename):
    st.download_button(label="Download Summary", data=content, file_name=filename, mime="text/plain")

# Main function
def main():
    st.set_page_config(layout="wide")
    
    # Create sidebar for language selection
    st.sidebar.title("Language Selection")
    language_code = st.sidebar.selectbox(
        "Select Language",
        list(languages.values()),
        format_func=lambda code: [lang for lang, lang_code in languages.items() if lang_code == code][0]
    )
    
    st.title("Text Summarization and Translation App")
    st.sidebar.title("Options")
    choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize URL", "Summarize Document", "Summarize Text from Clipboard"])
    
    # Define session state variables for clearing inputs
    if 'text' not in st.session_state:
        st.session_state.text = ""
    if 'url' not in st.session_state:
        st.session_state.url = ""
    if 'clipboard_text' not in st.session_state:
        st.session_state.clipboard_text = ""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    # Handle each choice
    if choice == "Summarize Text":
        st.session_state.text = st.text_area("Enter Text", st.session_state.text)
        maxlength = st.slider("Maximum Summary Length", min_value=50, max_value=1000, value=200)
        
        # Buttons for clearing input and history
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Clear Input"):
                clear_input(choice)
                st.experimental_rerun()
                
        with col2:
            if st.button("Clear History"):
                os.remove("summary_history.txt") if os.path.exists("summary_history.txt") else None
                st.experimental_rerun()

        if st.button("Summarize"):
            if validate_input(st.session_state.text):
                with st.spinner("Processing..."):
                    text = preprocess_text(st.session_state.text)
                    summary = text_summary(text, maxlength)
                    sentiment = sentiment_analysis(text)
                    if language_code != "en":
                        translated_summary = translate_text(summary, target_language=language_code)
                    else:
                        translated_summary = summary
                    
                    # Display sentiment analysis and summary
                    st.write("### Sentiment Analysis")
                    st.write(f"Label: {sentiment['label']}")
                    st.write(f"Score: {sentiment['score']:.2f}")

                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    elif choice == "Summarize URL":
        url = st.text_input("Enter URL", st.session_state.url)
        st.session_state.url = url
        
        if st.button("Summarize URL"):
            if validate_input(url):
                with st.spinner("Processing..."):
                    text = extract_text_from_url(url)
                    if text:
                        text = preprocess_text(text)
                        summary = text_summary(text)
                        sentiment = sentiment_analysis(text)
                        if language_code != "en":
                            translated_summary = translate_text(summary, target_language=language_code)
                        else:
                            translated_summary = summary

                        # Display sentiment analysis and summary
                        st.write("### Sentiment Analysis")
                        st.write(f"Label: {sentiment['label']}")
                        st.write(f"Score: {sentiment['score']:.2f}")

                        st.write("### Summary")
                        st.write(translated_summary)

                        save_summary(translated_summary)
                        download_file(translated_summary, "summary.txt")

    elif choice == "Summarize Document":
        uploaded_files = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "html", "csv", "xml", "jpg", "jpeg", "png"], accept_multiple_files=True)
        st.session_state.uploaded_files = uploaded_files
        
        if st.button("Summarize Documents"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    all_texts = ""
                    for uploaded_file in uploaded_files:
                        if uploaded_file.type == "application/pdf":
                            text = extract_text_from_pdf(uploaded_file)
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            text = extract_text_from_docx(uploaded_file)
                        elif uploaded_file.type == "text/plain":
                            text = extract_text_from_txt(uploaded_file)
                        elif uploaded_file.type == "text/html":
                            text = extract_text_from_html(uploaded_file)
                        elif uploaded_file.type == "text/csv":
                            text = extract_text_from_csv(uploaded_file)
                        elif uploaded_file.type == "application/xml":
                            text = extract_text_from_xml(uploaded_file)
                        elif uploaded_file.type in ["image/jpeg", "image/png"]:
                            text = extract_text_from_image(uploaded_file)
                        else:
                            text = ""
                        
                        all_texts += preprocess_text(text) + " "
                    
                    if all_texts:
                        summary = text_summary(all_texts)
                        sentiment = sentiment_analysis(all_texts)
                        if language_code != "en":
                            translated_summary = translate_text(summary, target_language=language_code)
                        else:
                            translated_summary = summary

                        # Display sentiment analysis and summary
                        st.write("### Sentiment Analysis")
                        st.write(f"Label: {sentiment['label']}")
                        st.write(f"Score: {sentiment['score']:.2f}")

                        st.write("### Summary")
                        st.write(translated_summary)

                        save_summary(translated_summary)
                        download_file(translated_summary, "summary.txt")

    elif choice == "Summarize Text from Clipboard":
        clipboard_text = st.text_area("Paste clipboard text here", st.session_state.clipboard_text)
        st.session_state.clipboard_text = clipboard_text

        if st.button("Summarize Clipboard"):
            if validate_input(clipboard_text):
                with st.spinner("Processing..."):
                    text = preprocess_text(clipboard_text)
                    summary = text_summary(text)
                    sentiment = sentiment_analysis(text)
                    if language_code != "en":
                        translated_summary = translate_text(summary, target_language=language_code)
                    else:
                        translated_summary = summary

                    # Display sentiment analysis and summary
                    st.write("### Sentiment Analysis")
                    st.write(f"Label: {sentiment['label']}")
                    st.write(f"Score: {sentiment['score']:.2f}")

                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    # Display summary history
    st.subheader("Summary History")
    history = load_summary_history()
    if history:
        st.write(history)
    else:
        st.write("No history available.")

if __name__ == "__main__":
    main()
