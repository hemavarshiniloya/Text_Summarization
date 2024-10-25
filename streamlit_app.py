import streamlit as st
import nltk
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import PyPDF2
import docx
import os

# Function to download necessary NLTK and TextBlob data
def download_required_data():
    """Download necessary NLTK and TextBlob corpora."""
    # Download NLTK data
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('brown')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    # Download TextBlob corpora
    try:
        from textblob import download_corpora
        download_corpora.download_all()
    except Exception as e:
        st.error(f"Error downloading TextBlob corpora: {str(e)}")

# Function to extract text from a given URL
def extract_text_from_url(url):
    """Extract text from a given URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return ' '.join(p.get_text() for p in soup.find_all('p'))
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return ""

# Function to extract text from a PDF file
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

# Function to extract text from a Word document
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

# Function to summarize the given text using TextBlob
def summarize_text(text, num_sentences=3):
    """Summarize the given text using TextBlob."""
    if not text:
        return ""
    
    try:
        blob = TextBlob(text)
        sentences = blob.sentences
        sorted_sentences = sorted(sentences, key=lambda s: len(s.noun_phrases), reverse=True)
        summary_sentences = [str(sentence) for sentence in sorted_sentences[:num_sentences]]
        summary = ' '.join(summary_sentences)
        return summary
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        return ""

# Main function for the Streamlit app
def main():
    st.title("📝 Text Summarization App")
    
    # Download required data at startup
    with st.spinner("Downloading required data..."):
        download_required_data()
        st.success("Required data downloaded successfully!")

    st.markdown("---")

    choice = st.sidebar.radio("Choose an option", ["Summarize Text", "Summarize URL", "Summarize Document"])

    if choice == "Summarize Text":
        input_text = st.text_area("Enter your text here:", height=300)
        num_sentences = st.number_input("Number of sentences for summary:", min_value=1, max_value=10, value=3)

        if st.button("Summarize"):
            if input_text:
                summary = summarize_text(input_text, num_sentences)
                st.write("### Summary")
                st.write(summary)
            else:
                st.warning("Please enter some text to summarize.")

    elif choice == "Summarize URL":
        url = st.text_input("Enter URL:")
        num_sentences = st.number_input("Number of sentences for summary:", min_value=1, max_value=10, value=3)

        if st.button("Summarize"):
            if url:
                text_from_url = extract_text_from_url(url)
                if text_from_url:
                    summary = summarize_text(text_from_url, num_sentences)
                    st.write("### Summary")
                    st.write(summary)
                else:
                    st.write("No text found at the provided URL.")
            else:
                st.warning("Please enter a URL to summarize.")

    elif choice == "Summarize Document":
        uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
        num_sentences = st.number_input("Number of sentences for summary:", min_value=1, max_value=10, value=3)

        if st.button("Summarize"):
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    text_from_document = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text_from_document = extract_text_from_docx(uploaded_file)

                if text_from_document:
                    summary = summarize_text(text_from_document, num_sentences)
                    st.write("### Summary")
                    st.write(summary)
                else:
                    st.write("No text found in the uploaded document.")
            else:
                st.warning("Please upload a document to summarize.")

if __name__ == "__main__":
    main()
