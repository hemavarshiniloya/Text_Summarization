import streamlit as st
import nltk
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document
import io
import subprocess
import sys
import os

# Function to download NLTK and TextBlob data
def download_nlp_resources():
    """Download required NLTK and TextBlob data."""
    try:
        # Create data directory if it doesn't exist
        if not os.path.exists("nltk_data"):
            os.makedirs("nltk_data")
        
        # Download NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        # Download TextBlob corpora
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "textblob.download_corpora"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return True
    except Exception as e:
        st.error(f"Error downloading NLP resources: {str(e)}")
        return False

# Initialize NLTK and TextBlob data
@st.cache_resource
def initialize_nlp():
    """Initialize NLP resources."""
    return download_nlp_resources()

# Text extraction from URL
def extract_text_from_url(url):
    """Extract text content from a given URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text().strip() for p in paragraphs])
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from URL: {str(e)}")
        return ""

# PDF text extraction
def extract_text_from_pdf(file):
    """Extract text content from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# DOCX text extraction
def extract_text_from_docx(file):
    """Extract text content from a DOCX file."""
    try:
        doc = Document(file)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

# Text summarization
def summarize_text(text, num_sentences=3):
    """Generate a summary of the input text."""
    try:
        # Create TextBlob object
        blob = TextBlob(text)
        sentences = blob.sentences
        
        if not sentences:
            return "No sentences found in the text."
        
        # Score sentences based on noun phrases
        sentence_scores = []
        for sentence in sentences:
            noun_phrases = sentence.noun_phrases
            score = len(noun_phrases)
            sentence_scores.append((sentence, score))
        
        # Sort sentences by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences
        top_sentences = [s[0] for s in sentence_scores[:num_sentences]]
        
        # Sort by original position
        top_sentences.sort(key=lambda s: sentences.index(s))
        
        # Join sentences into summary
        summary = ' '.join(str(sentence) for sentence in top_sentences)
        return summary
    
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return ""

# Save summary
def save_summary(summary, filename="summary.txt"):
    """Create a download button for the summary."""
    try:
        st.download_button(
            label="Download Summary",
            data=summary,
            file_name=filename,
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"Error creating download button: {str(e)}")

def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="Text Summarization App",
        page_icon="📚",
        layout="wide"
    )
    
    # Title and description
    st.title("📚 Text Summarization App")
    st.markdown("""
    This app helps you create concise summaries from:
    - Text input
    - Web pages (URL)
    - PDF documents
    - Word documents (DOCX)
    """)
    
    # Initialize NLP resources
    with st.spinner("Initializing NLP resources..."):
        if not initialize_nlp():
            st.error("Failed to initialize NLP resources. Please try reloading the app.")
            return
    
    # Sidebar
    st.sidebar.title("Settings")
    input_method = st.sidebar.radio(
        "Choose Input Method:",
        ["Text Input", "URL", "Upload Document"]
    )
    
    # Main content
    if input_method == "Text Input":
        text_input = st.text_area(
            "Enter your text here:",
            height=300,
            placeholder="Paste or type your text here..."
        )
        
        num_sentences = st.slider(
            "Number of sentences in summary:",
            min_value=1,
            max_value=10,
            value=3
        )
        
        if st.button("Generate Summary", key="text_summary"):
            if text_input:
                with st.spinner("Generating summary..."):
                    summary = summarize_text(text_input, num_sentences)
                    if summary:
                        st.subheader("Summary")
                        st.write(summary)
                        save_summary(summary)
            else:
                st.warning("Please enter some text to summarize.")
    
    elif input_method == "URL":
        url_input = st.text_input(
            "Enter URL:",
            placeholder="https://example.com"
        )
        
        num_sentences = st.slider(
            "Number of sentences in summary:",
            min_value=1,
            max_value=10,
            value=3
        )
        
        if st.button("Generate Summary", key="url_summary"):
            if url_input:
                with st.spinner("Fetching and summarizing content..."):
                    text = extract_text_from_url(url_input)
                    if text:
                        summary = summarize_text(text, num_sentences)
                        if summary:
                            st.subheader("Summary")
                            st.write(summary)
                            save_summary(summary)
                    else:
                        st.error("Could not extract text from URL.")
            else:
                st.warning("Please enter a URL.")
    
    else:  # Upload Document
        uploaded_file = st.file_uploader(
            "Upload document",
            type=["pdf", "docx"],
            help="Upload a PDF or Word document"
        )
        
        num_sentences = st.slider(
            "Number of sentences in summary:",
            min_value=1,
            max_value=10,
            value=3 )
        
        if st.button("Generate Summary", key="document_summary"):
            if uploaded_file:
                with st.spinner("Extracting and summarizing content..."):
                    if uploaded_file.type == "application/pdf":
                        text = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        text = extract_text_from_docx(uploaded_file)
                    
                    if text:
                        summary = summarize_text(text, num_sentences)
                        if summary:
                            st.subheader("Summary")
                            st.write(summary)
                            save_summary(summary)
                    else:
                        st.error("Could not extract text from document.")
            else:
                st.warning("Please upload a document.")

if __name__ == "__main__":
    main()
