import sys
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings

def load_environment():
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY is not set in the .env file or environment.")
        sys.exit(1)

def ingest_to_chroma(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()

    # Initialize Chroma with persistence
    chroma_settings = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory='./chroma_db'  # This directory will store the persisted data
    )

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        client_settings=chroma_settings,
        persist_directory='./chroma_db'
    )

    # Persistence is now handled automatically
    print(f"Text from {file_path} has been successfully ingested into Chroma and persisted.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_text_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)

    load_environment()
    ingest_to_chroma(file_path)