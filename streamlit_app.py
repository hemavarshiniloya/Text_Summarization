import streamlit as st
import nltk
from nltk.data import find
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import xml.etree.ElementTree as ET
import os
from googletrans import Translator
import re

# Download necessary NLTK data files
nltk.download("punkt", quiet=True)

# Ensure the required tokenizer is downloaded
try:
    find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# List of languages with their ISO 639-1 codes
languages = {
    "English": "en",
    "Afrikaans": "af",
    "Albanian": "sq",
    "Arabic": "ar",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Basque": "eu",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese (Simplified)": "zh-cn",
    "Chinese (Traditional)": "zh-tw",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "Georgian": "ka",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Haitian Creole": "ht",
    "Hebrew": "iw",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Igbo": "ig",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jw",
    "Kazakh": "kk",
    "Korean": "ko",
    "Kurdish": "ku",
    "Kyrgyz": "ky",
    "Lao": "lo",
    "Latin": "la",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Luxembourgish": "lb",
    "Macedonian": "mk",
    "Malagasy": "mg",
    "Malay": "ms",
    "Malayalam": "ml",
    "Maltese": "mt",
    "Maori": "mi",
    "Marathi": "mr",
    "Mongolian": "mn",
    "Nepali": "ne",
    "Norwegian": "no",
    "Pashto": "ps",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian": "sr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Xhosa": "xh",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "Zulu": "zu",
}

# Set page configuration
st.set_page_config(layout="wide")

def text_summary(text, max_sentences):
    """Summarize the given text using LSA summarizer."""
    if not text:
        raise ValueError("Input text cannot be empty.")
    
    # Use a try-except block to handle potential LookupError when creating Tokenizer
    try:
        tokenizer = Tokenizer("english")
    except LookupError:
        st.error("Tokenizer for English not found. Please ensure NLTK data is installed.")
        return ""
    
    parser = PlaintextParser.from_string(text, tokenizer)
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, max_sentences)
    return " ".join(str(sentence) for sentence in summary)

def preprocess_text(text):
    """Preprocess the input text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def extract_text_from_url(url):
    """Extract text from a URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        text = "\n".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        st.error(f"An error occurred while extracting text from URL: {str(e)}")
        return ""

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file):
    """Extract text from a TXT file."""
    return file.read().decode("utf-8")

def extract_text_from_html(file):
    """Extract text from an HTML file."""
    soup = BeautifulSoup(file.read(), "html.parser")
    paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    text = "\n".join([p.get_text() for p in paragraphs])
    return text

def extract_text_from_csv(file):
    """Extract text from a CSV file."""
    df = pd.read_csv(file)
    return df.to_string()

def extract_text_from_xml(file):
    """Extract text from an XML file."""
    tree = ET.parse(file)
    root = tree.getroot()
    text = " ".join([elem.text for elem in root.iter() if elem.text])
    return text

def save_summary(summary):
    """Save summary to history."""
    filename = "summary_history.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(summary + "\n\n")

def load_summary_history():
    """Load summary history."""
    filename = "summary_history.txt"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def clear_summary_history():
    """Clear summary history."""
    filename = "summary_history.txt"
    if os.path.exists(filename):
        os.remove(filename)

def clear_input(choice):
    """Clear input fields based on choice."""
    if choice == "Summarize Text":
        st.session_state.text = ""
    elif choice == "Summarize URL":
        st.session_state.url = ""
    elif choice == "Summarize Document":
        st.session_state.uploaded_files = []
    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = ""

def validate_input(text):
    """Validate input text."""
    return bool(text and text.strip())

def download_file(content, filename):
    """Download file functionality."""
    st.download_button(label="Download Summary", data=content, file_name=filename, mime="text/plain")

def main():
    st.title("Text Summarization App")

    # Initialize session state attributes if they don't exist
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

    # Text Summarization Section
    if choice == "Summarize Text":
        st.session_state.text = st.text_area("Enter your text here:", value=st.session_state.text, height=300)
        max_sentences = st.number_input("Maximum number of sentences for summary", min_value=1, value=2, step=1)

        if st.button("Summarize"):
            if validate_input(st.session_state.text):
                preprocessed_text = preprocess_text(st.session_state.text)
                summary = text_summary(preprocessed_text, max_sentences)
                st.success("Summary:")
                st.write(summary)
                save_summary(summary)
            else:
                st.error("Please enter valid text.")

    # URL Summarization Section
    elif choice == "Summarize URL":
        st.session_state.url = st.text_input("Enter the URL here:", value=st.session_state.url)

        if st.button("Summarize"):
            if validate_input(st.session_state.url):
                extracted_text = extract_text_from_url(st.session_state.url)
                if extracted_text:
                    preprocessed_text = preprocess_text(extracted_text)
                    summary = text_summary(preprocessed_text, 2)
                    st.success("Summary:")
                    st.write(summary)
                    save_summary(summary)
                else:
                    st.error("No text found at the provided URL.")
            else:
                st.error("Please enter a valid URL.")

    # Document Summarization Section
    elif choice == "Summarize Document":
        uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx", "txt", "csv", "xml", "html"], accept_multiple_files=True)

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

            for uploaded_file in st.session_state.uploaded_files:
                if uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_text_from_docx(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    text = extract_text_from_txt(uploaded_file)
                elif uploaded_file.type == "text/html":
                    text = extract_text_from_html(uploaded_file)
                elif uploaded_file.type == "application/xml":
                    text = extract_text_from_xml(uploaded_file)
                elif uploaded_file.type == "text/csv":
                    text = extract_text_from_csv(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    continue

                if text:
                    preprocessed_text = preprocess_text(text)
                    summary = text_summary(preprocessed_text, 2)
                    st.success(f"Summary of {uploaded_file.name}:")
                    st.write(summary)
                    save_summary(summary)

    # Clipboard Summarization Section
    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = st.text_area("Paste your text here:", value=st.session_state.clipboard_text, height=300)

        if st.button("Summarize"):
            if validate_input(st.session_state.clipboard_text):
                preprocessed_text = preprocess_text(st.session_state.clipboard_text)
                summary = text_summary(preprocessed_text, 2)
                st.success("Summary:")
                st.write(summary)
                save_summary(summary)
            else:
                st.error("Please enter valid text.")

    # History Section
    if st.sidebar.button("View Summary History"):
        history = load_summary_history()
        if history:
            st.text_area("Summary History", history, height=300)
        else:
            st.info("No summary history found.")

    # Clear History Section
    if st.sidebar.button("Clear Summary History"):
        clear_summary_history()
        st.success("Summary history cleared.")

    # Clear Input Section
    if st.sidebar.button("Clear Input"):
        clear_input(choice)
        st.success("Input cleared.")

if __name__ == "__main__":
    main()
