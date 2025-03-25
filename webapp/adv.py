import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import chromadb
import os

# Set up ChromaDB
CHROMA_DB_DIR = "./chroma_db"  # Set to None for in-memory

def fetch_content(url):
    """Extracts text content from a webpage using Selenium and BeautifulSoup."""
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--user-agent=Mozilla/5.0")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()

        content = '\n'.join([p.get_text() for p in soup.find_all("p")])
        return content if content else None
    except Exception:
        return None

def process_content(text):
    """Splits long texts into chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100, add_start_index=True)
    return text_splitter.split_text(text)

def main():
    st.set_page_config(page_title="Smart Q&A", layout="wide")
    st.title("Smart Q&A with AI")
    st.write("Enter a URL to get insights!")

    url = st.text_input("Enter URL:")
    question = st.text_input("Ask a question about the content:")

    if st.button("Get Answer") and url and question:
        with st.spinner("Fetching content..."):
            text = fetch_content(url)
            if not text:
                st.error("Failed to fetch content. Try another URL.")
                return

        with st.spinner("Processing content..."):
            all_splits = process_content(text)
            if not all_splits:
                st.error("No relevant text found.")
                return
        
        with st.spinner("Generating answer..."):
            embeddings = OllamaEmbeddings(model="llama3:8b")

            # Ensure proper ChromaDB handling
            if CHROMA_DB_DIR:
                os.makedirs(CHROMA_DB_DIR, exist_ok=True)

            vectorstore = Chroma.from_texts(
                texts=all_splits, 
                embedding=embeddings, 
                persist_directory=CHROMA_DB_DIR
            )

            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(question)

            # Extract text from Document objects
            context = ' '.join([doc.page_content for doc in retrieved_docs if isinstance(doc, Document)])

            if not context.strip():
                st.error("No relevant content found to answer the question.")
                return

            llm = OllamaLLM(model="llama3:8b")
            response = llm.invoke(f"Answer based on the context:\n\nQuestion: {question}\n\nContext: {context}")

        st.subheader("Answer:")
        st.write(response)

if __name__ == "__main__":
    main()
