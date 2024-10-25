import streamlit as st
from textblob import TextBlob
import nltk
import requests
from bs4 import BeautifulSoup
import PyPDF2
import docx
from docx import Document
import os
from datetime import datetime
import base64
from fpdf import FPDF
import io

# Add these new functions for saving summaries

def save_as_txt(summary, filename="summary.txt"):
    """Save summary as TXT file and return download link."""
    try:
        # Create buffer
        buffer = io.StringIO()
        buffer.write(summary)
        buffer.seek(0)
        
        # Create download button
        btn = st.download_button(
            label="Download as TXT",
            data=buffer.getvalue(),
            file_name=filename,
            mime="text/plain"
        )
        return btn
    except Exception as e:
        st.error(f"Error saving as TXT: {str(e)}")
        return None

def save_as_pdf(summary, filename="summary.pdf"):
    """Save summary as PDF file and return download link."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Split text into lines that fit the PDF width
        lines = [summary[i:i+90] for i in range(0, len(summary), 90)]
        for line in lines:
            pdf.cell(0, 10, txt=line, ln=True)
        
        # Save to buffer
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        
        # Create download button
        btn = st.download_button(
            label="Download as PDF",
            data=pdf_buffer,
            file_name=filename,
            mime="application/pdf"
        )
        return btn
    except Exception as e:
        st.error(f"Error saving as PDF: {str(e)}")
        return None

def save_as_docx(summary, filename="summary.docx"):
    """Save summary as DOCX file and return download link."""
    try:
        # Create document
        doc = Document()
        doc.add_paragraph(summary)
        
        # Save to buffer
        docx_buffer = io.BytesIO()
        doc.save(docx_buffer)
        docx_buffer.seek(0)
        
        # Create download button
        btn = st.download_button(
            label="Download as DOCX",
            data=docx_buffer,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        return btn
    except Exception as e:
        st.error(f"Error saving as DOCX: {str(e)}")
        return None

def display_download_options(summary, original_text=""):
    """Display download options for the summary."""
    st.markdown("### Download Options")
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        save_as_txt(summary, f"summary_{timestamp}.txt")
    
    with col2:
        save_as_pdf(summary, f"summary_{timestamp}.pdf")
    
    with col3:
        save_as_docx(summary, f"summary_{timestamp}.docx")

    # Option to save both original and summary
    if original_text:
        st.markdown("### Save Original Text with Summary")
        combined_text = f"Original Text:\n\n{original_text}\n\nSummary:\n\n{summary}"
        save_as_txt(combined_text, f"full_text_and_summary_{timestamp}.txt")

# Modify your main function to include saving options
def main():
    st.title("Text Summarization App")

    choice = st.sidebar.radio("Choose an option", ["Summarize Text", "Summarize URL", "Summarize Document"])

    if choice == "Summarize Text":
        input_text = st.text_area("Enter your text here:", height=300)
        num_sentences = st.number_input("Number of sentences for summary:", min_value=1, max_value=10, value=3)

        if st.button("Summarize"):
            if input_text:
                summary = summarize_text(input_text, num_sentences)
                st.write("### Summary")
                st.write(summary)
                
                # Display download options
                display_download_options(summary, input_text)
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
                    
                    # Display download options
                    display_download_options(summary, text_from_url)
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
                else:
                    text_from_document = extract_text_from_docx(uploaded_file)

                if text_from_document:
                    summary = summarize_text(text_from_document, num_sentences)
                    st.write("### Summary")
                    st.write(summary)
                    
                    # Display download options
                    display_download_options(summary, text_from_document)
                else:
                    st.write("No text found in the uploaded document.")
            else:
                st.warning("Please upload a document to summarize.")

if __name__ == "__main__":
    main()
