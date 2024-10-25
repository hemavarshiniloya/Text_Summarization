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
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

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

# Initialize the LSA summarizer
summarizer = LsaSummarizer()

def text_summary(text, max_sentences):
    # Parse the input text
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    # Generate the summary
    summary = summarizer(parser.document, max_sentences)
    
    # Convert summary to string
    return " ".join(str(sentence) for sentence in summary)

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
    
    # Summarization algorithm selection
    algorithm = "LSA"  # Fixed to LSA

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

    # Text Summarization Section
    if choice == "Summarize Text":
        st.session_state.text = st.text_area("Enter your text here:", value=st.session_state.text, height=300)
        max_sentences = st.number_input("Maximum number of sentences for summary", min_value=1, value=2, step=1)

        if st.button("Summarize"):
            if validate_input(st.session_state.text):
                preprocessed_text = preprocess_text(st.session_state.text)
                summary = text_summary(preprocessed_text, max_sentences)
                st.write("### Summary")
                st.write(summary)
                
                # Save summary to history
                save_summary(summary)
                
                # Download summary
                download_file(summary, "summary.txt")

            else:
                st.error("Please enter valid text.")

        if st.button("Clear Input"):
            clear_input("Summarize Text")

    # URL Summarization Section
    elif choice == "Summarize URL":
        st.session_state.url = st.text_input("Enter URL:", value=st.session_state.url)

        if st.button("Summarize"):
            if validate_input(st.session_state.url):
                text = extract_text_from_url(st.session_state.url)
                if text:
                    max_sentences = st.number_input("Maximum number of sentences for summary", min_value=1, value=2, step=1)
                    summary = text_summary(preprocess_text(text), max_sentences)
                    st.write("### Summary")
                    st.write(summary)

                    # Save summary to history
                    save_summary(summary)

                    # Download summary
                    download_file(summary, "summary.txt")

                else:
                    st.error("No text could be extracted from the URL.")
            else:
                st.error("Please enter a valid URL.")

        if st.button("Clear Input"):
            clear_input("Summarize URL")

    # Document Summarization Section
    elif choice == "Summarize Document":
        uploaded_files = st.file_uploader("Upload Document (PDF/DOCX/TXT/HTML/CSV/XML)", accept_multiple_files=True, type=["pdf", "docx", "txt", "html", "csv", "xml"])

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

        if st.button("Summarize"):
            if st.session_state.uploaded_files:
                for file in st.session_state.uploaded_files:
                    file_type = file.name.split('.')[-1].lower()

                    if file_type == "pdf":
                        text = extract_text_from_pdf(file)
                    elif file_type == "docx":
                        text = extract_text_from_docx(file)
                    elif file_type == "txt":
                        text = extract_text_from_txt(file)
                    elif file_type == "html":
                        text = extract_text_from_html(file)
                    elif file_type == "csv":
                        text = extract_text_from_csv(file)
                    elif file_type == "xml":
                        text = extract_text_from_xml(file)
                    else:
                        st.error(f"Unsupported file type: {file_type}")
                        continue

                    if text:
                        max_sentences = st.number_input("Maximum number of sentences for summary", min_value=1, value=2, step=1)
                        summary = text_summary(preprocess_text(text), max_sentences)
                        st.write(f"### Summary for {file.name}")
                        st.write(summary)

                        # Save summary to history
                        save_summary(summary)

                        # Download summary
                        download_file(summary, f"{file.name}_summary.txt")

            else:
                st.error("Please upload a document.")

        if st.button("Clear Input"):
            clear_input("Summarize Document")

    # Clipboard Summarization Section
    
    max_sentences = 5  # Default value, can be changed by user input
    
    
    max_sentences_input = st.number_input("Select max sentences for summary:", min_value=1, value=max_sentences)

    # Your other code
    # Line 363: Assuming you have a choice mechanism
    if choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = st.text_area("Paste your text here:", st.session_state.clipboard_text, key="clipboard_text_area")

        if st.button("Summarize Clipboard Text"):
            if validate_input(st.session_state.clipboard_text):
                preprocessed_text = preprocess_text(st.session_state.clipboard_text)
                summary = text_summary(preprocessed_text, max_sentences_input)  # Use the input variable
                st.write("### Summary:")
                st.write(summary)
                save_summary(summary)
                download_file(summary, "summary.txt")
            else:
                st.error("Please paste some text to summarize.")


        if st.button("Summarize"):
            if validate_input(st.session_state.clipboard_text):
                preprocessed_text = preprocess_text(st.session_state.clipboard_text)
                summary = text_summary(preprocessed_text, 2)  # Default to 2 sentences
                st.write("### Summary")
                st.write(summary)

                # Save summary to history
                save_summary(summary)

                # Download summary
                download_file(summary, "summary.txt")

            else:
                st.error("Please enter valid text.")

        if st.button("Clear Input"):
            clear_input("Summarize Text from Clipboard")

    # Summary History Section
    if st.sidebar.button("Show Summary History"):
        st.subheader("Summary History")
        history = load_summary_history()
        if history:
            st.write(history)
        else:
            st.write("No summary history available.")

    # Clear Summary History Button
    if st.sidebar.button("Clear Summary History"):
        clear_summary_history()
        st.success("Summary history cleared.")

# Run the app
if __name__ == "__main__":
    main()
