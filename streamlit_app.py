import streamlit as st
import spacy
import requests
from bs4 import BeautifulSoup
import PyPDF2
import docx
from collections import Counter
from heapq import nlargest
import io

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

class TextSummarizer:
    def __init__(self):
        self.nlp = load_spacy_model()

    def summarize(self, text, num_sentences=3):
        # Preprocess and clean the text
        doc = self.nlp(text)
        
        # Calculate word frequencies
        word_freq = Counter()
        for word in doc:
            if not word.is_stop and not word.is_punct and word.text.strip():
                word_freq[word.text.lower()] += 1

        # Normalize frequencies
        max_freq = max(word_freq.values()) if word_freq else 1
        normalized_freq = {word: freq/max_freq for word, freq in word_freq.items()}

        # Score sentences
        sent_strength = {}
        for sent in doc.sents:
            for word in sent:
                if word.text.lower() in normalized_freq:
                    if sent in sent_strength:
                        sent_strength[sent] += normalized_freq[word.text.lower()]
                    else:
                        sent_strength[sent] = normalized_freq[word.text.lower()]

        # Get top sentences
        summarized_sentences = nlargest(num_sentences, sent_strength, key=sent_strength.get)
        summary = ' '.join([str(sent) for sent in summarized_sentences])
        
        return summary

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs])
    except Exception as e:
        st.error(f"Error extracting text from URL: {str(e)}")
        return None

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

def main():
    st.set_page_config(
        page_title="Text Summarizer",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Text Summarizer")
    st.write("Upload text, URL, or documents to generate a summary.")

    # Initialize summarizer
    summarizer = TextSummarizer()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Text", "URL", "Document"])

    with tab1:
        st.header("Text Input")
        text_input = st.text_area("Enter your text here:", height=200)
        num_sentences = st.slider("Number of sentences in summary:", min_value=1, max_value=10, value=3)
        
        if st.button("Summarize Text", key="text_button"):
            if text_input:
                with st.spinner("Generating summary..."):
                    summary = summarizer.summarize(text_input, num_sentences)
                    st.success("Summary generated!")
                    st.write(summary)
            else:
                st.warning("Please enter some text to summarize.")

    with tab2:
        st.header("URL Input")
        url_input = st.text_input("Enter URL:")
        url_sentences = st.slider("Number of sentences in summary:", min_value=1, max_value=10, value=3, key="url_slider")
        
        if st.button("Summarize URL", key="url_button"):
            if url_input:
                with st.spinner("Fetching content and generating summary..."):
                    text = extract_text_from_url(url_input)
                    if text:
                        summary = summarizer.summarize(text, url_sentences)
                        st.success("Summary generated!")
                        st.write(summary)
                    else:
                        st.error("Could not extract text from the URL.")
            else:
                st.warning("Please enter a URL.")

    with tab3:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'docx'])
        doc_sentences = st.slider("Number of sentences in summary:", min_value=1, max_value=10, value=3, key="doc_slider")
        
        if uploaded_file and st.button("Summarize Document", key="doc_button"):
            with st.spinner("Processing document and generating summary..."):
                file_type = uploaded_file.type
                if file_type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_text_from_docx(uploaded_file)
                else:
                    text = None
                    st.error("Unsupported file type")

                if text:
                    summary = summarizer.summarize(text, doc_sentences)
                    st.success("Summary generated!")
                    st.write(summary)

    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center">
            <p>Made with ❤️ by Your Hema Varshini Loya</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
