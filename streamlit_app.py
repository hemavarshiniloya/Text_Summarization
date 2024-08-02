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
from googletrans import Translator  # Use Google Translate API
from textblob import TextBlob  # For sentiment analysis
import re

# List of languages with their ISO 639-1 codes
languages = {
    "English": "en", 
    "Afrikaans": "af",
    "Albanian": "sq",
    # (Include all languages here)
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Xhosa": "xh",
    "Yoruba": "yo",
    "Zulu": "zu"
}

# Set page configuration
st.set_page_config(layout="wide")

# Initialize text summarizer
def text_summary(text, maxlength=None):
    summary = Summary()
    result = summary(text)
    return result

# Initialize sentiment analyzer
def sentiment_analysis(text):
    try:
        sentiment_pipeline = hf_pipeline("sentiment-analysis")
        result = sentiment_pipeline(text)
        return result
    except Exception as e:
        st.error(f"An error occurred during sentiment analysis: {str(e)}")
        return [{"label": "ERROR", "score": 0.0}]

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
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Function to download file
def download_file(content, filename):
    st.download_button(label="Download Summary", data=content, file_name=filename, mime="text/plain")

# Main function
def main():
    st.title("Summarization App")
    st.sidebar.title("Options")
    choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize URL", "Summarize Document", "Summarize Text from Clipboard"])

    # Set English as the default language
    default_language_code = "en"
    language_code = st.sidebar.selectbox(
        "Select Language",
        list(languages.values()),
        format_func=lambda x: [k for k, v in languages.items() if v == x][0],
        index=list(languages.values()).index(default_language_code)
    )

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

        if st.button("Summarize"):
            if validate_input(st.session_state.text):
                with st.spinner("Processing..."):
                    text = preprocess_text(st.session_state.text)
                    summary = text_summary(text, maxlength)
                    sentiment = sentiment_analysis(text)
                    if language_code != default_language_code:
                        translated_summary = translate_text(summary, target_language=language_code)
                    else:
                        translated_summary = summary
                    
                    # Display sentiment analysis and summary
                    st.write("### Sentiment Analysis")
                    st.write(f"Label: {sentiment[0]['label']}, Score: {sentiment[0]['score']:.2f}")

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
                        if language_code != default_language_code:
                            translated_summary = translate_text(summary, target_language=language_code)
                        else:
                            translated_summary = summary

                        # Display sentiment analysis and summary
                        st.write("### Sentiment Analysis")
                        st.write(f"Label: {sentiment[0]['label']}, Score: {sentiment[0]['score']:.2f}")

                        st.write("### Summary")
                        st.write(translated_summary)

                        save_summary(translated_summary)
                        download_file(translated_summary, "summary.txt")

    elif choice == "Summarize Document":
        uploaded_files = st.file_uploader("Choose files", type=["pdf", "docx", "txt", "html", "csv", "xml"], accept_multiple_files=True)
        st.session_state.uploaded_files = uploaded_files

        if uploaded_files:
            all_texts = ""
            for uploaded_file in uploaded_files:
                file_type = uploaded_file.type
                if file_type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_text_from_docx(uploaded_file)
                elif file_type == "text/plain":
                    text = extract_text_from_txt(uploaded_file)
                elif file_type == "text/html":
                    text = extract_text_from_html(uploaded_file)
                elif file_type == "text/csv":
                    text = extract_text_from_csv(uploaded_file)
                elif file_type == "application/xml":
                    text = extract_text_from_xml(uploaded_file)
                else:
                    st.error(f"Unsupported file type: {file_type}")
                    continue

                all_texts += preprocess_text(text) + "\n\n"

            if st.button("Summarize Document"):
                if validate_input(all_texts):
                    with st.spinner("Processing..."):
                        summary = text_summary(all_texts)
                        sentiment = sentiment_analysis(all_texts)
                        if language_code != default_language_code:
                            translated_summary = translate_text(summary, target_language=language_code)
                        else:
                            translated_summary = summary

                        # Display sentiment analysis and summary
                        st.write("### Sentiment Analysis")
                        st.write(f"Label: {sentiment[0]['label']}, Score: {sentiment[0]['score']:.2f}")

                        st.write("### Summary")
                        st.write(translated_summary)

                        save_summary(translated_summary)
                        download_file(translated_summary, "summary.txt")

    elif choice == "Summarize Text from Clipboard":
        clipboard_text = st.text_area("Paste Clipboard Text", st.session_state.clipboard_text)
        st.session_state.clipboard_text = clipboard_text

        if st.button("Summarize Clipboard Text"):
            if validate_input(clipboard_text):
                with st.spinner("Processing..."):
                    text = preprocess_text(clipboard_text)
                    summary = text_summary(text)
                    sentiment = sentiment_analysis(text)
                    if language_code != default_language_code:
                        translated_summary = translate_text(summary, target_language=language_code)
                    else:
                        translated_summary = summary

                    # Display sentiment analysis and summary
                    st.write("### Sentiment Analysis")
                    st.write(f"Label: {sentiment[0]['label']}, Score: {sentiment[0]['score']:.2f}")

                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    # Sidebar button to clear input
    if st.sidebar.button("Clear Input"):
        clear_input(choice)

    # Sidebar button to view summary history
    if st.sidebar.button("View Summary History"):
        history = load_summary_history()
        if history:
            st.write("### Summary History")
            st.write(history)
        else:
            st.write("No summary history available.")

if __name__ == "__main__":
    main()
