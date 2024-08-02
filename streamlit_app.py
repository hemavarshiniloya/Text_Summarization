import streamlit as st
from txtai.pipeline import Summary
from transformers import pipeline as hf_pipeline
from googletrans import Translator
from textblob import TextBlob
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
from openpyxl import load_workbook
from ebooklib import epub
from pptx import Presentation
import pandas as pd
import xml.etree.ElementTree as ET
import pytesseract
import requests
import re
import os

# List of languages with their ISO 639-1 codes
languages = {
    "English": "en", 
    "Afrikaans": "af",
    "Albanian": "sq",
    "Amharic": "am",
    "Arabic": "ar",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Basque": "eu",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese (Simplified)": "zh",
    "Chinese (Traditional)": "zh-TW",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "Esperanto": "eo",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "Georgian": "ka",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Haitian Creole": "ht",
    "Hausa": "ha",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Igbo": "ig",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Khmer": "km",
    "Kinyarwanda": "rw",
    "Korean": "ko",
    "Kurdish": "ku",
    "Kyrgyz": "ky",
    "Lao": "lo",
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
    "Samoan": "sm",
    "Scots Gaelic": "gd",
    "Serbian": "sr",
    "Sesotho": "st",
    "Shona": "sn",
    "Sindhi": "sd",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Spanish": "es",
    "Sundanese": "su",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tajik": "tg",
    "Tamil": "ta",
    "Tatar": "tt",
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
    sentiment_pipeline = hf_pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)
    return result

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

# Function to extract text from Excel (XLSX)
def extract_text_from_excel(file):
    workbook = load_workbook(file)
    text = ""
    for sheet in workbook.sheetnames:
        worksheet = workbook[sheet]
        for row in worksheet.iter_rows(values_only=True):
            row_text = " ".join([str(cell) for cell in row if cell is not None])
            text += row_text + "\n"
    return text

# Function to extract text from EPUB
def extract_text_from_epub(file):
    book = epub.read_epub(file)
    text = ""
    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text += soup.get_text()
    return text

# Function to extract text from PowerPoint (PPTX)
def extract_text_from_pptx(file):
    presentation = Presentation(file)
    text = ""
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
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

# Function to validate input fields
def validate_input(choice):
    if choice == "Summarize Text" and not st.session_state.text:
        return False
    elif choice == "Summarize URL" and not st.session_state.url:
        return False
    elif choice == "Summarize Document" and not st.session_state.uploaded_files:
        return False
    elif choice == "Summarize Text from Clipboard" and not st.session_state.clipboard_text:
        return False
    return True

# Function to translate text
def translate_text(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

# Function to download summary as a file
def download_file(text, filename):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Summary</a>'
    st.markdown(href, unsafe_allow_html=True)

# Sidebar for input choice
st.sidebar.title("Input Choice")
input_choice = st.sidebar.radio(
    "Choose the input type:", 
    ("Summarize Text", "Summarize URL", "Summarize Document", "Summarize Text from Clipboard")
)

# Sidebar for language choice
st.sidebar.title("Language Choice")
default_language_code = languages["English"]
selected_language = st.sidebar.selectbox("Select language for output summary:", list(languages.keys()))
language_code = languages[selected_language]

# Main app logic based on input choice
if input_choice == "Summarize Text":
    st.title("Summarize Text")
    text_input = st.text_area("Enter text to summarize:", key="text")
    
    if st.button("Summarize"):
        if validate_input(input_choice):
            with st.spinner("Summarizing..."):
                text_input = preprocess_text(text_input)
                summary = text_summary(text_input)
                sentiment = sentiment_analysis(text_input)
                if language_code != default_language_code:
                    translated_summary = translate_text(summary, target_language=language_code)
                else:
                    translated_summary = summary

                st.write("### Sentiment Analysis")
                st.write(f"Score: {sentiment[0]['score']:.2f}")

                st.write("### Summary")
                st.write(translated_summary)

                save_summary(translated_summary)
                download_file(translated_summary, "summary.txt")

    if st.button("Clear Input"):
        clear_input(input_choice)

elif input_choice == "Summarize URL":
    st.title("Summarize URL")
    url_input = st.text_input("Enter URL to summarize:", key="url")
    
    if st.button("Summarize"):
        if validate_input(input_choice):
            with st.spinner("Extracting and summarizing..."):
                text = extract_text_from_url(url_input)
                if text:
                    text = preprocess_text(text)
                    summary = text_summary(text)
                    sentiment = sentiment_analysis(text)
                    if language_code != default_language_code:
                        translated_summary = translate_text(summary, target_language=language_code)
                    else:
                        translated_summary = summary

                    st.write("### Sentiment Analysis")
                    st.write(f"Score: {sentiment[0]['score']:.2f}")

                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    if st.button("Clear Input"):
        clear_input(input_choice)

elif input_choice == "Summarize Document":
    st.title("Summarize Document")
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=["pdf", "docx", "txt", "html", "csv", "xml", "xlsx", "epub", "pptx"], 
        accept_multiple_files=True,
        key="uploaded_files"
    )
    
    if st.button("Summarize"):
        if validate_input(input_choice):
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
                elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    text = extract_text_from_excel(uploaded_file)
                elif file_type == "application/epub+zip":
                    text = extract_text_from_epub(uploaded_file)
                elif file_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
                    text = extract_text_from_pptx(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    continue

                if text:
                    all_texts += text + "\n"

            if all_texts:
                with st.spinner("Processing..."):
                    all_texts = preprocess_text(all_texts)
                    summary = text_summary(all_texts)
                    sentiment = sentiment_analysis(all_texts)
                    if language_code != default_language_code:
                        translated_summary = translate_text(summary, target_language=language_code)
                    else:
                        translated_summary = summary

                    st.write("### Sentiment Analysis")
                    st.write(f"Score: {sentiment[0]['score']:.2f}")

                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    if st.button("Clear Input"):
        clear_input(input_choice)

elif input_choice == "Summarize Text from Clipboard":
    st.title("Summarize Text from Clipboard")
    clipboard_text_input = st.text_area("Paste text from clipboard:", key="clipboard_text")
    
    if st.button("Summarize"):
        if validate_input(input_choice):
            with st.spinner("Summarizing..."):
                clipboard_text_input = preprocess_text(clipboard_text_input)
                summary = text_summary(clipboard_text_input)
                sentiment = sentiment_analysis(clipboard_text_input)
                if language_code != default_language_code:
                    translated_summary = translate_text(summary, target_language=language_code)
                else:
                    translated_summary = summary

                st.write("### Sentiment Analysis")
                st.write(f"Score: {sentiment[0]['score']:.2f}")

                st.write("### Summary")
                st.write(translated_summary)

                save_summary(translated_summary)
                download_file(translated_summary, "summary.txt")

    if st.button("Clear Input"):
        clear_input(input_choice)

# Display history of summaries
st.sidebar.title("Summary History")
summary_history = load_summary_history()
st.sidebar.text_area("Summary History", summary_history, height=200)
