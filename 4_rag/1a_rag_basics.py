import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    try:
        loader = TextLoader(file_path, encoding="utf-8")  # Specify encoding
        documents = loader.load()
    except UnicodeDecodeError as e:
        print(f"Error decoding file: {e}")
        print("Trying 'latin-1' encoding...")
        try:
            loader = TextLoader(file_path, encoding="latin-1")
            documents = loader.load()
        except UnicodeDecodeError as e2:
            print(f"Error decoding with 'latin-1' as well: {e2}")
            print("Please investigate the file encoding.  You may need to try other encodings like 'utf-16', 'ascii' with errors='ignore' (lossy).")
            raise  # Re-raise the exception to stop execution


    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[10].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    #embeddings = OpenAIEmbeddings( model="text-embedding-3-small" )  # Update to a valid embedding model if needed
    embeddings = HuggingFaceEmbeddings(model_name="all-minilm-l6-v2")
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
