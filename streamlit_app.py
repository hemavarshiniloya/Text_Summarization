import streamlit as st
import nltk
from nltk.data import find
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import xml.etree.ElementTree as ET
import os
from googletrans import Translator
import re
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
    "English": "en",
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

# Initialize text summarizer
def text_summary(text, maxlength=None):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()  # Using TextRank summarizer
    summary = summarizer(parser.document, maxlength)  # Length is in sentences
    return ' '.join(str(sentence) for sentence in summary)

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
        st.error(f"An error occurred while extracting text from URL: {str(e)}")
        return ""

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

# Main function to run the Streamlit app
def main():
    st.title("Text Summarization App")

    # Language selection
    selected_language = st.sidebar.selectbox("Select Language", options=list(languages.keys()), index=0)

    # Handle choice selection
    choice = st.sidebar.radio("Choose an option", ["Summarize Text", "Summarize URL", "Summarize Document", "Summarize Text from Clipboard"])

    # Initialize session state attributes if they don't exist
    if 'text' not in st.session_state:
        st.session_state.text = ""
    if 'url' not in st.session_state:
        st.session_state.url = ""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'clipboard_text' not in st.session_state:
        st.session_state.clipboard_text = ""

    # Handle each choice
    if choice == "Summarize Text":
        st.session_state.text = st.text_area("Enter Text", st.session_state.text)
        maxlength = st.slider("Maximum Summary Length (sentences)", 1, 10, 5)

        if st.button("Summarize"):
            if validate_input(st.session_state.text):
                preprocessed_text = preprocess_text(st.session_state.text)
                summary = text_summary(preprocessed_text, maxlength)
                st.write("Summary:")
                st.write(summary)
                save_summary(summary)
            else:
                st.error("Please enter some text to summarize.")

    elif choice == "Summarize URL":
        st.session_state.url = st.text_input("Enter URL", st.session_state.url)

        if st.button("Extract and Summarize"):
            if validate_input(st.session_state.url):
                extracted_text = extract_text_from_url(st.session_state.url)
                preprocessed_text = preprocess_text(extracted_text)
                summary = text_summary(preprocessed_text)
                st.write("Summary:")
                st.write(summary)
                save_summary(summary)
            else:
                st.error("Please enter a valid URL.")

    elif choice == "Summarize Document":
        uploaded_files = st.file_uploader("Upload Document", type=["txt", "pdf", "docx", "csv", "xml", "html"], accept_multiple_files=True)
        st.session_state.uploaded_files = uploaded_files if uploaded_files else st.session_state.uploaded_files

        if st.button("Summarize Documents"):
            all_text = ""
            for uploaded_file in st.session_state.uploaded_files:
                if uploaded_file.type == "application/pdf":
                    all_text += extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    all_text += extract_text_from_docx(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    all_text += extract_text_from_txt(uploaded_file)
                elif uploaded_file.type == "text/html":
                    all_text += extract_text_from_html(uploaded_file)
                elif uploaded_file.type == "text/csv":
                    all_text += extract_text_from_csv(uploaded_file)
                elif uploaded_file.type == "application/xml":
                    all_text += extract_text_from_xml(uploaded_file)

            if all_text:
                preprocessed_text = preprocess_text(all_text)
                summary = text_summary(preprocessed_text)
                st.write("Summary:")
                st.write(summary)
                save_summary(summary)
            else:
                st.error("No text found in the uploaded documents.")

    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = st.text_area("Text from Clipboard", st.session_state.clipboard_text)

        if st.button("Summarize Clipboard Text"):
            if validate_input(st.session_state.clipboard_text):
                preprocessed_text = preprocess_text(st.session_state.clipboard_text)
                summary = text_summary(preprocessed_text)
                st.write("Summary:")
                st.write(summary)
                save_summary(summary)
            else:
                st.error("Clipboard text is empty.")

    # Translation feature
    if st.button("Translate Summary"):
        if 'summary' in locals():
            translated_summary = translate_text(summary, languages[selected_language])
            st.write("Translated Summary:")
            st.write(translated_summary)
        else:
            st.error("Please summarize text before translating.")

    # Summary history
    if st.sidebar.button("Show Summary History"):
        history = load_summary_history()
        if history:
            st.sidebar.text_area("Summary History", history, height=300)
        else:
            st.sidebar.write("No summary history available.")

    # Clear summary history
    if st.sidebar.button("Clear Summary History"):
        clear_summary_history()
        st.sidebar.write("Summary history cleared.")

    # Clear inputs based on choice
    if st.sidebar.button("Clear Input"):
        clear_input(choice)

if __name__ == "__main__":
    main()
