import streamlit as st
import PyPDF2
import docx
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

class TextSummarizer:
    def __init__(self, language='english'):
        self.language = language
        # Initialize the summarization pipeline
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize(self, text, num_sentences=3):
        try:
            # Summarize the text
            summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            return f"An error occurred while summarizing: {str(e)}"

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return None

def extract_text_from_txt(txt_file):
    try:
        text = txt_file.read().decode("utf-8")
        return text
    except Exception as e:
        st.error(f"Error extracting text from TXT: {str(e)}")
        return None

def extract_text_from_excel(excel_file):
    try:
        df = pd.read_excel(excel_file, sheet_name=None)  # Read all sheets
        text = ""
        for sheet_name, sheet_data in df.items():
            text += f"Sheet: {sheet_name}\n"
            text += sheet_data.to_string(index=False) + "\n\n"  # Convert DataFrame to string
        return text
    except Exception as e:
        st.error(f"Error extracting text from Excel: {str(e)}")
        return None

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join(paragraph.get_text() for paragraph in paragraphs)
        return text
    except Exception as e:
        st.error(f"Error extracting text from URL: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Summarization",
        page_icon="📚",
        layout="wide"
    )

    st.title("Summarization")  # Updated title
    st.write("Upload multiple documents (PDF, DOCX, TXT, XLSX), input text directly, or enter a URL to generate summaries.")

    # Language selection
    language = st.selectbox("Select Language for Summarization:", ["English": "en",
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
    "Zulu": "zu",])  # Add more languages as needed
    summarizer = TextSummarizer(language)

    # File uploader for multiple files
    uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT, XLSX)", type=['pdf', 'docx', 'txt', 'xlsx'], accept_multiple_files=True)
    input_text = st.text_area("Input text directly:")
    input_url = st.text_input("Enter a URL:")

    num_sentences = st.slider("Number of sentences in summary:", min_value=1, max_value=10, value=3)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.subheader(f"Processing {uploaded_file.name} ...")
            
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(uploaded_file)
            elif uploaded_file.type == "text/plain":
                text = extract_text_from_txt(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                text = extract_text_from_excel(uploaded_file)
            else:
                st.error("Unsupported file type.")
                continue
            
            if text:
                st.write("Extracted Text:")
                st.write(text)
                summary = summarizer.summarize(text, num_sentences)
                st.write("Summary:")
                st.write(summary)
            else:
                st.error("Failed to extract text from the file.")

    if input_text:
        st.write("Input Text:")
        st.write(input_text)
        summary = summarizer.summarize(input_text, num_sentences)
        st.write("Summary:")
        st.write(summary)

    if input_url:
        st.write("Input URL:")
        st.write(input_url)
        text = extract_text_from_url(input_url)
        if text:
            st.write("Extracted Text:")
            st.write(text)
            summary = summarizer.summarize(text, num_sentences)
            st.write("Summary:")
            st.write(summary)
        else:
            st.error("Failed to extract text from the URL.")

if __name__ == "__main__":
    main()
