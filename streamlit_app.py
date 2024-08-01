import streamlit as st
from txtai.pipeline import Summary
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import xml.etree.ElementTree as ET
import os
import pytesseract
from googletrans import Translator  # Use Google Translate API

# Set Tesseract executable path if necessary
# Uncomment and set the correct path if you are running locally
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# List of languages with their ISO 639-1 codes
languages = {
    "Afrikaans": "af", "Albanian": "sq", "Amharic": "am", "Arabic": "ar",
    "Armenian": "hy", "Azerbaijani": "az", "Basque": "eu", "Belarusian": "be",
    "Bengali": "bn", "Bosnian": "bs", "Bulgarian": "bg", "Catalan": "ca",
    "Chinese (Simplified)": "zh", "Chinese (Traditional)": "zh-TW", "Croatian": "hr",
    "Czech": "cs", "Danish": "da", "Dutch": "nl", "English": "en",
    "Esperanto": "eo", "Estonian": "et", "Finnish": "fi", "French": "fr",
    "Galician": "gl", "Georgian": "ka", "German": "de", "Greek": "el",
    "Gujarati": "gu", "Haitian Creole": "ht", "Hausa": "ha", "Hebrew": "he",
    "Hindi": "hi", "Hungarian": "hu", "Icelandic": "is", "Igbo": "ig",
    "Indonesian": "id", "Irish": "ga", "Italian": "it", "Japanese": "ja",
    "Javanese": "jv", "Kannada": "kn", "Kazakh": "kk", "Khmer": "km",
    "Kinyarwanda": "rw", "Korean": "ko", "Kurdish": "ku", "Kyrgyz": "ky",
    "Lao": "lo", "Latvian": "lv", "Lithuanian": "lt", "Luxembourgish": "lb",
    "Macedonian": "mk", "Malagasy": "mg", "Malay": "ms", "Malayalam": "ml",
    "Maltese": "mt", "Maori": "mi", "Marathi": "mr", "Mongolian": "mn",
    "Nepali": "ne", "Norwegian": "no", "Pashto": "ps", "Persian": "fa",
    "Polish": "pl", "Portuguese": "pt", "Punjabi": "pa", "Romanian": "ro",
    "Russian": "ru", "Samoan": "sm", "Scots Gaelic": "gd", "Serbian": "sr",
    "Sesotho": "st", "Shona": "sn", "Sindhi": "sd", "Sinhala": "si",
    "Slovak": "sk", "Slovenian": "sl", "Somali": "so", "Spanish": "es",
    "Sundanese": "su", "Swahili": "sw", "Swedish": "sv", "Tagalog": "tl",
    "Tajik": "tg", "Tamil": "ta", "Tatar": "tt", "Telugu": "te",
    "Thai": "th", "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur",
    "Uzbek": "uz", "Vietnamese": "vi", "Welsh": "cy", "Xhosa": "xh",
    "Yoruba": "yo", "Zulu": "zu"
}

# Set page configuration
st.set_page_config(layout="wide")

# Initialize text summarizer
@st.cache_resource
def text_summary(text, maxlength=None):
    summary = Summary()
    result = summary(text)
    return result

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

# Function to clear all inputs and outputs
def clear_all():
    st.session_state.clear()
    # Clear summary history file
    history_file = "summary_history.txt"
    if os.path.exists(history_file):
        os.remove(history_file)

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

    if choice == "Summarize Text":
        text = st.text_area("Enter Text", "")
        maxlength = st.slider("Maximum Summary Length", min_value=50, max_value=1000, value=200)

        if st.button("Summarize"):
            if validate_input(text):
                summary = text_summary(text, maxlength)
                if language_code != default_language_code:
                    translated_summary = translate_text(summary, target_language=language_code)
                else:
                    translated_summary = summary
                st.write("### Summary")
                st.write(translated_summary)
                save_summary(translated_summary)
                download_file(translated_summary, "summary.txt")

    elif choice == "Summarize URL":
        url = st.text_input("Enter URL", "")

        if st.button("Summarize URL"):
            if validate_input(url):
                text = extract_text_from_url(url)
                if text:
                    summary = text_summary(text)
                    if language_code != default_language_code:
                        translated_summary = translate_text(summary, target_language=language_code)
                    else:
                        translated_summary = summary
                    st.write("### Summary")
                    st.write(translated_summary)
                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    elif choice == "Summarize Document":
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "html", "csv", "xml"])

        if uploaded_file is not None:
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
                st.error("Unsupported file type.")
                text = None

            if text:
                summary = text_summary(text)
                if language_code != default_language_code:
                    translated_summary = translate_text(summary, target_language=language_code)
                else:
                    translated_summary = summary
                st.write("### Summary")
                st.write(translated_summary)
                save_summary(translated_summary)
                download_file(translated_summary, "summary.txt")

    elif choice == "Summarize Text from Clipboard":
        clipboard_text = st.text_area("Paste text from clipboard", "")

        if st.button("Summarize Clipboard Text"):
            if validate_input(clipboard_text):
                summary = text_summary(clipboard_text)
                if language_code != default_language_code:
                    translated_summary = translate_text(summary, target_language=language_code)
                else:
                    translated_summary = summary
                st.write("### Summary")
                st.write(translated_summary)
                save_summary(translated_summary)
                download_file(translated_summary, "summary.txt")

    st.sidebar.button("Clear All", on_click=clear_all)
    st.sidebar.button("Clear Summary History", on_click=lambda: os.remove("summary_history.txt") if os.path.exists("summary_history.txt") else None)

    # Display summary history
    st.write("### Summary History")
    history = load_summary_history()
    st.text_area("Previously Summarized Texts", history, height=300)

if __name__ == "__main__":
    main()
