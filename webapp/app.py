import streamlit as st
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma

def fetch_content(url):
    """Extract content from the given URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}  # Prevent blocking
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None  # Invalid URL or blocked request
        
        loader = WebBaseLoader(web_paths=[url])
        docs = loader.load()
        return docs
    except Exception as e:
        return None

def process_content(docs):
    """Split the extracted content for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100, add_start_index=True)
    return text_splitter.split_documents(docs)

def main():
    st.set_page_config(page_title="Smart Q&A with AI", layout="wide")
    st.title("Smart Q&A with AI")
    st.write("Enter a URL to get insights!")
    
    url = st.text_input("Enter URL:")
    question = st.text_input("Ask a question about the content:")
    
    if st.button("Get Answer") and url and question:
        docs = fetch_content(url)
        if not docs:
            st.error("Failed to fetch content. Try another URL.")
            return
        
        all_splits = process_content(docs)
        embeddings = OllamaEmbeddings(model="llama3:8b")
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(question)
        context = ' '.join([doc.page_content for doc in retrieved_docs])
        
        llm = OllamaLLM(model="llama3:8b")
        response = llm.invoke(f"""Answer briefly based on the context:
            Question: {question}.
            Context: {context}
        """)
        
        st.subheader("Answer:")
        st.write(response)
        
        st.success("Open this app in any browser by running `streamlit run app.py` in your terminal.")

if __name__ == "__main__":
    main()
