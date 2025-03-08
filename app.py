import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.getenv("GEMINI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Extract text from PDFs


def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF documents."""
    text_list = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text_list.extend(
            page.extract_text() for page in pdf_reader.pages
            if page.extract_text()
        )
    return "\n".join(text_list)

# Split text into chunks


def get_text_chunks(text):
    """Splits extracted text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )
    return text_splitter.split_text(text)

# Convert text chunks into vector embeddings


def get_vectors(txt_chunks):
    """Converts text chunks into vector embeddings and saves them."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    vector_store = FAISS.from_texts(txt_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Define conversational prompt


def get_conversational_prompt():
    """Defines the conversational prompt for answering user queries."""
    prompt_template = (
        "Answer the questions as detailed as possible from the "
        "provided content. If you don’t know the answer, simply "
        "say: 'Answer not available in the context'\n\nContext:\n{context}\n\n"
        "Question:\n{question}\n\nAnswer:\n"
    )
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro", temperature=0.3
    )
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user queries


def user_input(user_query):
    """
    Handles user queries by retrieving relevant text chunks
    and generating a response using the conversational model.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    try:
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_query)
        chain = get_conversational_prompt()
        response = chain(
            {"input_document": docs, "question": user_query},
            return_complete=True
        )
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit UI


def main():
    """Defines the Streamlit UI and handles user interactions."""
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.header("Chat with Multiple PDFs using Gemini 💁")
    user_question = st.text_input("Ask a Question from the PDFs:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload PDF files:", accept_multiple_files=True, type=["pdf"]
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text found in the uploaded PDFs.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vectors(text_chunks)
                        st.success("Processing Complete ✅")


if __name__ == "__main__":
    main()
