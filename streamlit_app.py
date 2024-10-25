import streamlit as st
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import PyPDF2
import docx
import nltk

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('brown')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

def extract_text_from_url(url):
    """Extract text from a given URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return ' '.join(p.get_text() for p in soup.find_all('p'))
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return ""

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def extract_text_from_docx(file):
    """Extract text from a Word document."""
    try:
        doc = docx.Document(file)
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text
    except Exception as e:
        st.error(f"Error reading Word document: {e}")
        return ""

def summarize_text(text, num_sentences=3):
    """Summarize the given text using TextBlob."""
    try:
        # Ensure NLTK data is downloaded
        if not download_nltk_data():
            st.error("Failed to download required NLTK data")
            return ""

        blob = TextBlob(text)
        sentences = blob.sentences
        
        # Sort sentences by importance (using noun phrases as a metric)
        sorted_sentences = sorted(sentences, key=lambda s: len(s.noun_phrases), reverse=True)
        
        # Get the top n sentences
        summary = ' '.join(str(sentence) for sentence in sorted_sentences[:num_sentences])
        return summary
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return ""

def main():
    st.title("Text Summarization App")
    
    # Initialize NLTK data at startup
    with st.spinner("Initializing required data..."):
        download_nltk_data()
    
    choice = st.sidebar.radio("Choose an option", ["Summarize Text", "Summarize URL", "Summarize Document"])

    if choice == "Summarize Text":
        input_text = st.text_area("Enter your text here:", height=300)
        num_sentences = st.number_input("Number of sentences for summary:", min_value=1, max_value=10, value=3)

        if st.button("Summarize"):
            if input_text:
                with st.spinner("Generating summary..."):
                    summary = summarize_text(input_text, num_sentences)
                    if summary:
                        st.write("### Summary")
                        st.write(summary)
            else:
                st.warning("Please enter some text to summarize.")

    elif choice == "Summarize URL":
        url = st.text_input("Enter URL:")
        num_sentences = st.number_input("Number of sentences for summary:", min_value=1, max_value=10, value=3)

        if st.button("Summarize"):
            if url:
                with st.spinner("Extracting text from URL..."):
                    text_from_url = extract_text_from_url(url)
                    if text_from_url:
                        with st.spinner("Generating summary..."):
                            summary = summarize_text(text_from_url, num_sentences)
                            if summary:
                                st.write("### Summary")
                                st.write(summary)
                    else:
                        st.warning("No text found at the provided URL.")
            else:
                st.warning("Please enter a URL to summarize.")

    elif choice == "Summarize Document":
        uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
        num_sentences = st.number_input("Number of sentences for summary:", min_value=1, max_value=10, value=3)

        if st.button("Summarize"):
            if uploaded_file:
                with st.spinner("Extracting text from document..."):
                    if uploaded_file.type == "application/pdf":
                        text_from_document = extract_text_from_pdf(uploaded_file)
                    else:
                        text_from_document = extract_text_from_docx(uploaded_file)

                    if text_from_document:
                        with st.spinner("Generating summary..."):
                            summary = summarize_text(text_from_document, num_sentences)
                            if summary:
                                st.write("### Summary")
                                st.write(summary)
                    else:
                        st.warning("No text found in the uploaded document.")
            else:
                st.warning("Please upload a document to summarize.")

if __name__ == "__main__":
    main()
