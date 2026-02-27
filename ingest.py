import os
import streamlit as st 
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. Configuration
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
KNOWLEDGE_DIR = "./myknowledge"  # Create this folder and put .txt files in it
DB_DIR = "./stockfish_db"

def build_kb():
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        return

    # 2. Load and Split
    loader = DirectoryLoader(KNOWLEDGE_DIR, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    
    # Adept-level tip: Using smaller chunks (500-1000) helps the AI find precise technical facts
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    # 3. Create Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print(f"database created")

if __name__ == "__main__":
    build_kb()