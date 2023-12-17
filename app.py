import streamlit as st
import PyPDF2
from transformers import pipeline
from huggingface_hub import cached_download
import tempfile

# Download and load pre-trained QA model
model = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')


def answer_question(pdf_path, question):

    # Extract text from PDF
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        page_content = ""
        for page in pdf_reader.pages:
            page_content += page.extract_text()

    # Use Hugging Face model for QA
    answer = model(question=question, context=page_content)
    return answer


def main():
    st.title("Safety Chatbot")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload PDF")

    # Initialize questions and answers list in session state
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # Ask questions
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            st.write(f"File uploaded: {uploaded_file.name}")

            question = st.text_input("Ask your question about the PDF:")

            if question:
                answer = answer_question(temp_file.name, question)
                st.write(f"Answer: {answer['answer']}")
                st.write("---")
                st.session_state.qa_history.append({"question": question, "answer": answer['answer']})

        # Display question and answer history
        st.subheader("Chat History:")
        for qa_pair in st.session_state.qa_history:
            st.write(f"You: {qa_pair['question']}")
            st.write(f"Answer: {qa_pair['answer']}")
            st.write("---")


if __name__ == "__main__":
    main()
