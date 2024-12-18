import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings  # Updated import for CohereEmbeddings
from langchain_community.llms import Cohere
import json  # To handle data in JSON format

# Function to set environment variables for Cohere and Pinecone
def set_environment_variables():
    """Set API keys for Cohere and Pinecone."""
    os.environ["COHERE_API_KEY"] = "V5jqL0gNm1xo9gIzN0x32c9Nc814b4gh9gpOY6EY"
    os.environ["PINECONE_API_KEY"] = "pcsk_22X17E_TyjHKBUTtPwKSdfG7dMkcnCGHqPBShmgd9cugqxdFmVuBXYxWrnEgXDV4id7Stq"

# Function to initialize Cohere embeddings
def initialize_embeddings():
    """Initialize Cohere embeddings."""
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY is not set in the environment variables.")
    
    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=cohere_api_key,  # Ensure API key is passed
        user_agent="my_custom_app"  # Add user_agent to prevent KeyError
    )

# Function to split text into manageable chunks
def split_text_into_chunks(docs, chunk_size=1500, chunk_overlap=200):
    """
    Split text documents into manageable chunks.
    
    Args:
        docs (list): List of text documents.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
    
    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(docs)

# Function to create a vector store from the text chunks
def create_vector_store(chunks, embeddings, index_name):
    """
    Create a new vector store with the given chunks.
    
    Args:
        chunks (list): List of text chunks.
        embeddings: Embedding model to convert text into vectors.
        index_name (str): Name of the Pinecone index.
    """
    PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=index_name
    )

# Function to add documents to an existing vector store
# Function to add documents to an existing vector store
def add_docs_to_vector_store(docs, embeddings, index_name):
    """
    Add new documents to an existing vector store.
    
    Args:
        docs (list): List of documents (e.g., rows from CSV).
        embeddings: Embedding model.
        index_name (str): Name of the Pinecone index.
    """
    vector_store = PineconeVectorStore.from_existing_index(index_name, embeddings)
    
    # Split each document into chunks and add to vector store
    for doc in docs:
        chunks = split_text_into_chunks(doc)  # Pass doc directly as a string, not in a list
        vector_store.add_texts(chunks)

# Main function
def main():
    # Set environment variables
    set_environment_variables()

    # Initialize embeddings
    embeddings = initialize_embeddings()  # Initialize embeddings once

    # Read CSV files from the folder
    folder_path = 'D:\\TVS\\shakthi\\anime'  # Replace with the correct path
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Process each CSV file
    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        print(f"Processing file: {csv_path}")
        
        # Load CSV data
        df = pd.read_csv(csv_path)

        # Loop through each row and treat the entire row as a complete document
        docs = []
        for _, row in df.iterrows():
            # Combine all columns' values into a single string (or JSON object)
            document = " ".join([str(value) for value in row])  # Combine all column values as a string
            
            # Convert to JSON (or list, depending on your needs)
            document_json = json.dumps({"columns": row.to_dict()})  # Using JSON representation of the entire row
            
            docs.append(document_json)  # Add the document as JSON

        # Add to Pinecone vector store
        index_name = "tvs"  # Use the existing index name
        add_docs_to_vector_store(docs, embeddings, index_name)

    print("All documents have been processed and added to the vector store.")

# Run the main function
if __name__ == "__main__":
    main()
