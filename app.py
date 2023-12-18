import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import tempfile

# Download and load pre-trained QA model
model = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')


def answer_question(pdf_texts, question):
    # Concatenate the text from all PDFs
    combined_text = "\n".join(pdf_texts)

    # Use Hugging Face model for QA
    answer = model(question=question, context=combined_text)
    return answer


def main():
    st.title("Safety Chatbot")

    # Upload multiple PDF files
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

    # Initialize questions and answers list in session state
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # Store text from each PDF
    pdf_texts = []

    # Ask general questions
    question = st.text_input("Ask your general question:")

    # Process uploaded files
    if uploaded_files:
        for file_idx, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())

                # Extract text from PDF using PyMuPDF
                pdf_document = fitz.open(temp_file.name)
                page_content = ""
                for page_number in range(pdf_document.page_count):
                    page = pdf_document[page_number]
                    page_text = page.get_text("text")
                    page_content += page_text

                pdf_texts.append(page_content)

        # Display general question and answer
        if question:
            answer = answer_question(pdf_texts, question)
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
