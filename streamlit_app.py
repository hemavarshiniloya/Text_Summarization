import streamlit as st
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import PyPDF2
import docx
import nltk
import os

class TextSummarizer:
    def __init__(self):
        self.initialize_nlp()

    @staticmethod
    def initialize_nlp():
        """Initialize NLTK and TextBlob data"""
        required_nltk_data = ['punkt', 'averaged_perceptron_tagger', 'brown', 'wordnet', 'omw-1.4']
        for item in required_nltk_data:
            try:
                nltk.data.find(f'tokenizers/{item}')
            except LookupError:
                nltk.download(item)

    def extract_text_from_url(self, url):
        """Extract text from a given URL"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            return ' '.join(p.get_text() for p in soup.find_all('p'))
        except Exception as e:
            st.error(f"Error fetching URL: {e}")
            return ""

    def extract_text_from_pdf(self, file):
        """Extract text from a PDF file"""
        try:
            reader = PyPDF2.PdfReader(file)
            return ' '.join(page.extract_text() or '' for page in reader.pages)
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return ""

    def extract_text_from_docx(self, file):
        """Extract text from a Word document"""
        try:
            doc = docx.Document(file)
            return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as e:
            st.error(f"Error reading Word document: {e}")
            return ""

    def summarize_text(self, text, num_sentences=3):
        """Summarize the given text"""
        try:
            blob = TextBlob(text)
            sentences = blob.sentences
            # Sort sentences by importance (number of noun phrases)
            sorted_sentences = sorted(sentences, key=lambda s: len(s.noun_phrases), reverse=True)
            return ' '.join(str(sentence) for sentence in sorted_sentences[:num_sentences])
        except Exception as e:
            st.error(f"Error summarizing text: {e}")
            return ""

class StreamlitApp:
    def __init__(self):
        self.summarizer = TextSummarizer()

    def run(self):
        st.title("Text Summarization App")
        
        choice = st.sidebar.radio(
            "Choose an option",
            ["Summarize Text", "Summarize URL", "Summarize Document"]
        )

        if choice == "Summarize Text":
            self.text_summarization()
        elif choice == "Summarize URL":
            self.url_summarization()
        elif choice == "Summarize Document":
            self.document_summarization()

    def text_summarization(self):
        input_text = st.text_area("Enter your text here:", height=300)
        num_sentences = st.number_input(
            "Number of sentences for summary:",
            min_value=1,
            max_value=10,
            value=3
        )

        if st.button("Summarize"):
            if input_text:
                with st.spinner("Generating summary..."):
                    summary = self.summarizer.summarize_text(input_text, num_sentences)
                    if summary:
                        st.write("### Summary")
                        st.write(summary)
            else:
                st.warning("Please enter some text to summarize.")

    def url_summarization(self):
        url = st.text_input("Enter URL:")
        num_sentences = st.number_input(
            "Number of sentences for summary:",
            min_value=1,
            max_value=10,
            value=3
        )

        if st.button("Summarize"):
            if url:
                with st.spinner("Extracting text from URL..."):
                    text = self.summarizer.extract_text_from_url(url)
                    if text:
                        summary = self.summarizer.summarize_text(text, num_sentences)
                        if summary:
                            st.write("### Summary")
                            st.write(summary)
                    else:
                        st.warning("No text found at the provided URL.")
            else:
                st.warning("Please enter a URL to summarize.")

    def document_summarization(self):
        uploaded_file = st.file_uploader(
            "Upload a PDF or Word document",
            type=["pdf", "docx"]
        )
        num_sentences = st.number_input(
            "Number of sentences for summary:",
            min_value=1,
            max_value=10,
            value=3
        )

        if st.button("Summarize"):
            if uploaded_file:
                with st.spinner("Processing document..."):
                    if uploaded_file.type == "application/pdf":
                        text = self.summarizer.extract_text_from_pdf(uploaded_file)
                    else:
                        text = self.summarizer.extract_text_from_docx(uploaded_file)

                    if text:
                        summary = self.summarizer.summarize_text(text, num_sentences)
                        if summary:
                            st.write("### Summary")
                            st.write(summary)
                    else:
                        st.warning("No text found in the uploaded document.")
            else:
                st.warning("Please upload a document to summarize.")

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
