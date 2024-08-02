import streamlit as st
from txtai.pipeline import Summary
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
from PIL import Image
import language_tool_python
from rake_nltk import Rake
from textblob import TextBlob
import nltk
nltk.download('punkt')

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

# Initialize tools
summarizer = Summary()
similarity = TfidfVectorizer()

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
    elif choice == "Text Similarity Comparison":
        st.session_state.text1 = ""
        st.session_state.text2 = ""

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

# Function to summarize text
def text_summary(text, maxlength=None):
    result = summarizer(text)
    return result

# Function to compare text similarity
def text_similarity(text1, text2):
    # Vectorize the texts
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(vectors)
    similarity_score = cosine_sim[0][1]  # Get the similarity between the two texts

    return similarity_score

# Main function to run the Streamlit app
def main():
    st.title("Text Processing App")

    # Language selection
    selected_language = st.sidebar.selectbox("Select Language", options=list(languages.keys()), index=0)

    # Handle choice selection
    choice = st.sidebar.radio(
        "Choose an option",
        ["Summarize Text", "Summarize URL", "Summarize Document", "Summarize Text from Clipboard", "Text Similarity Comparison"]
    )

    # Initialize session state attributes if they don't exist
    if 'text' not in st.session_state:
        st.session_state.text = ""
    if 'url' not in st.session_state:
        st.session_state.url = ""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'clipboard_text' not in st.session_state:
        st.session_state.clipboard_text = ""
    if 'text1' not in st.session_state:
        st.session_state.text1 = ""
    if 'text2' not in st.session_state:
        st.session_state.text2 = ""

    # Handle each choice
    if choice == "Summarize Text":
        st.session_state.text = st.text_area("Enter Text", st.session_state.text)

        if st.button("Summarize"):
            if validate_input(st.session_state.text):
                with st.spinner("Summarizing..."):
                    summary = text_summary(st.session_state.text)
                    st.write("### Summary")
                    st.write(summary)

                    # Translate summary if different language is selected
                    if selected_language != "English":
                        with st.spinner("Translating..."):
                            translated_summary = translate_text(summary, languages[selected_language])
                            st.write("### Translated Summary")
                            st.write(translated_summary)

                    # Save summary to history
                    save_summary(summary)
            else:
                st.error("Please enter valid text.")

    elif choice == "Summarize URL":
        st.session_state.url = st.text_input("Enter URL", st.session_state.url)

        if st.button("Summarize"):
            if validate_input(st.session_state.url):
                with st.spinner("Extracting and summarizing..."):
                    text = extract_text_from_url(st.session_state.url)
                    if text:
                        summary = text_summary(text)
                        st.write("### Summary")
                        st.write(summary)

                        # Translate summary if different language is selected
                        if selected_language != "English":
                            with st.spinner("Translating..."):
                                translated_summary = translate_text(summary, languages[selected_language])
                                st.write("### Translated Summary")
                                st.write(translated_summary)

                        # Save summary to history
                        save_summary(summary)
            else:
                st.error("Please enter a valid URL.")

    elif choice == "Summarize Document":
        st.session_state.uploaded_files = st.file_uploader("Upload Document", accept_multiple_files=True)

        if st.session_state.uploaded_files:
            for uploaded_file in st.session_state.uploaded_files:
                file_type = uploaded_file.type
                text = ""

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
                elif file_type == "text/xml":
                    text = extract_text_from_xml(uploaded_file)
                elif file_type.startswith("image/"):
                    text = extract_text_from_image(uploaded_file)
                else:
                    st.error(f"Unsupported file type: {file_type}")
                    continue

                if validate_input(text):
                    with st.spinner(f"Summarizing {uploaded_file.name}..."):
                        summary = text_summary(text)
                        st.write(f"### Summary of {uploaded_file.name}")
                        st.write(summary)

                        # Translate summary if different language is selected
                        if selected_language != "English":
                            with st.spinner("Translating..."):
                                translated_summary = translate_text(summary, languages[selected_language])
                                st.write("### Translated Summary")
                                st.write(translated_summary)

                        # Save summary to history
                        save_summary(summary)
                else:
                    st.error(f"Could not extract valid text from {uploaded_file.name}")

    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = st.text_area("Paste Text Here", st.session_state.clipboard_text)

        if st.button("Summarize"):
            if validate_input(st.session_state.clipboard_text):
                with st.spinner("Summarizing..."):
                    summary = text_summary(st.session_state.clipboard_text)
                    st.write("### Summary")
                    st.write(summary)

                    # Translate summary if different language is selected
                    if selected_language != "English":
                        with st.spinner("Translating..."):
                            translated_summary = translate_text(summary, languages[selected_language])
                            st.write("### Translated Summary")
                            st.write(translated_summary)

                    # Save summary to history
                    save_summary(summary)
            else:
                st.error("Please enter valid text.")

    elif choice == "Text Similarity Comparison":
        st.session_state.text1 = st.text_area("Enter Text 1", st.session_state.text1)
        st.session_state.text2 = st.text_area("Enter Text 2", st.session_state.text2)

        if st.button("Compare"):
            if validate_input(st.session_state.text1) and validate_input(st.session_state.text2):
                with st.spinner("Calculating similarity..."):
                    similarity_score = text_similarity(st.session_state.text1, st.session_state.text2)
                    
                    # Display similarity score
                    st.write("### Similarity Score")
                    st.write(f"The similarity between the texts is {similarity_score:.2f}")

    if st.sidebar.button("Clear Input"):
        clear_input(choice)

    if st.sidebar.button("Clear Summary History"):
        clear_summary_history()

    if st.sidebar.button("Load Summary History"):
        history = load_summary_history()
        if history:
            st.write("### Summary History")
            st.write(history)
        else:
            st.write("No summary history found.")

if __name__ == "__main__":
    main()
