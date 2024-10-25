import streamlit as st
import nltk
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import PyPDF2
import docx
import os

# Download NLTK data at app startup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Cache the NLTK downloads
@st.cache_resource
def download_nltk_data():
    """Download and cache NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

# Function to extract text from a given URL
@st.cache_data
def extract_text_from_url(url):
    """Extract text from a given URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join(p.get_text() for p in paragraphs)
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return ""

# Function to extract text from PDF
def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# Function to extract text from Word document
def extract_text_from_docx(file):
    """Extract text from a Word document."""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading Word document: {str(e)}")
        return ""

# Function to summarize text
@st.cache_data
def summarize_text(text, num_sentences=3):
    """Summarize the given text."""
    try:
        # Ensure NLTK data is downloaded
        download_nltk_data()
        
        # Create TextBlob object
        blob = TextBlob(text)
        
        # Get sentences and sort by importance (using length of noun phrases as a simple metric)
        sentences = blob.sentences
        sentence_scores = [(sentence, len(sentence.noun_phrases)) for sentence in sentences]
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        
        # Select top n sentences
        summary_sentences = [str(sentence[0]) for sentence in sorted_sentences[:num_sentences]]
        summary = ' '.join(summary_sentences)
        
        return summary
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        return "Error occurred during summarization."

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Text Summarization App",
        page_icon="📝",
        layout="wide"
    )

    # Download NLTK data at startup
    download_nltk_data()

    st.title("📝 Text Summarization App")
    st.markdown("---")

    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Text Input", "URL Input", "Document Upload"])

    with tab1:
        st.header("Summarize Text")
        input_text = st.text_area("Enter your text here:", height=200)
        col1, col2 = st.columns([1, 3])
        with col1:
            num_sentences = st.number_input("Number of sentences:", 1, 10, 3)
        with col2:
            if st.button("Summarize Text", key="text_button"):
                if input_text:
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(input_text, num_sentences)
                        st.success("Summary generated!")
                        st.markdown("### Summary")
                        st.write(summary)
                else:
                    st.warning("Please enter some text to summarize.")

    with tab2:
        st.header("Summarize from URL")
        url = st.text_input("Enter URL:")
        col1, col2 = st.columns([1, 3])
        with col1:
            num_sentences = st.number_input("Number of sentences:", 1, 10, 3, key="url_sentences")
        with col2:
            if st.button("Summarize URL", key="url_button"):
                if url:
                    with st.spinner("Fetching and summarizing content..."):
                        text = extract_text_from_url(url)
                        if text:
                            summary = summarize_text(text, num_sentences)
                            st.success("Summary generated!")
                            st.markdown("### Summary")
                            st.write(summary)
                        else:
                            st.error("Could not extract text from the URL.")
                else:
                    st.warning("Please enter a URL.")

    with tab3:
        st.header("Summarize Document")
        uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
        col1, col2 = st.columns([1, 3])
        with col1:
            num_sentences = st.number_input("Number of sentences:", 1, 10, 3, key="doc_sentences")
        with col2:
            if st.button("Summarize Document", key="doc_button"):
                if uploaded_file:
                    with st.spinner("Processing document..."):
                        if uploaded_file.type == "application/pdf":
                            text = extract_text_from_pdf(uploaded_file)
                        else:
                            text = extract_text_from_docx(uploaded_file)
                        
                        if text:
                            summary = summarize_text(text, num_sentences)
                            st.success("Summary generated!")
                            st.markdown("### Summary")
                            st.write(summary)
                        else:
                            st.error("Could not extract text from the document.")
                else:
                    st.warning("Please upload a document.")

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and TextBlob")

if __name__ == "__main__":
    main()
