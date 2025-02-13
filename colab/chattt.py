
import streamlit as st
import pytesseract
from PIL import Image
import pdfplumber
import google.generativeai as genai

def analyse():
    # Initialize the Gemini API
    genai.configure(api_key="AIzaSyDoO3STDZb5ehCVaoJtYjdy4uo8Pww4cTw")  # Replace with your Gemini API key

    # Initialize the Gemini model
    gemini_model = genai.GenerativeModel('gemini-pro')

    # LangChain Prompt Template for Summary
    summary_template = (
        "You are an experienced virtual health assistant specializing in providing accurate medical information. "
        "Summarize the following medical report in no more than 300 words. "
        "At the end of the summary, clearly state if the health condition appears to be 'Good' or 'Needs Attention' based on the provided data. "
        "Do not suggest consulting a doctor:\n\n{context}"
        "tell whether a person's health condition is normal or not. "
        "If they don't have a normal condition, then only say contact doctor."
    )

    # LangChain Prompt Template for Q&A
    qa_template = (
        "You are a virtual health advisor providing detailed and self-sufficient medical insights. "
        "Based on the following report:\n\n{context}\n\nAnswer this question: {question}. "
        "Avoid recommending consulting a doctor. Instead, focus on self-care tips, over-the-counter advice, and home remedies."
    )

    # Function to clean response (remove unwanted suggestions)
    def clean_response(response):
        forbidden_phrases = [ "seek medical advice", "healthcare professional", "visit a physician"]
        for phrase in forbidden_phrases:
            response = response.replace(phrase, "")
        return response

    # Function to extract text from image using OCR
    def extract_text_from_image(image_path):
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text

    # Function to extract text from PDF
    def extract_text_from_pdf(pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text

    # Function to generate summary using Gemini model
    def generate_summary(context):
        prompt = summary_template.format(context=context)
        response = gemini_model.generate_content(prompt)
        return clean_response(response.text)

    # Function to get the answer from the model based on the extracted text
    def get_answer(question, context):
        prompt = qa_template.format(context=context, question=question)
        response = gemini_model.generate_content(prompt)
        return clean_response(response.text)

  

    # File Upload Section
    st.subheader("Upload Your Report File")
    uploaded_file = st.file_uploader("Choose a file (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Check file type and process accordingly
        if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            text = extract_text_from_image(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
            st.text_area("📋 Extracted Text from PDF", text, height=300)

        # Generate Summary Using Gemini
        if text:
            st.subheader("📝 Generated Summary")
            summary = generate_summary(text)
            st.write(summary)

        # Q&A Section
        st.subheader("❓ Ask a Question about the Report")
        question = st.text_input("Enter your question here")

        if question:
            if text:
                answer = get_answer(question, text)
                st.markdown("### ✅ Answer:")
                st.write(answer)
            else:
                st.warning("No text extracted from the file. Please upload a valid file.")


