import fitz  # PyMuPDF for extracting text from PDFs
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

# Initialize Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Vector database (using Faiss)
index = faiss.IndexFlatL2(384)  # 384 is the dimension of the sentence embeddings
stored_embeddings = []
stored_chunks = []

# Helper function to extract text from PDF
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")  # Extract text from each page
    return text

# Helper function to chunk the text (using simple sentence-level segmentation)
def chunk_text(text, chunk_size=500):
    # Split the text into chunks of specified size
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Helper function to generate embeddings for chunks
def generate_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings

def process_pdf(pdf_path):
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text)
    embeddings = generate_embeddings(chunks)
    
    # Store the embeddings in the Faiss index
    global stored_embeddings, stored_chunks
    stored_embeddings.extend(embeddings)
    stored_chunks.extend(chunks)
    
    # Add only new embeddings to Faiss index
    embeddings_np = np.array(embeddings).astype('float32')
    if embeddings_np.shape[1] == index.d:  # Ensure dimension matches
        index.add(embeddings_np)
    else:
        print(f"Error: Dimension mismatch. Index expects {index.d}, got {embeddings_np.shape[1]}")


# Query Handling: Convert query to embeddings and retrieve relevant chunks
def handle_query(query):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Perform similarity search in the vector database
    D, I = index.search(query_embedding, k=5)  # k is the number of nearest neighbors to retrieve
    
    # Extract the most relevant chunks
    relevant_chunks = [stored_chunks[i] for i in I[0]]
    
    # Combine the relevant chunks into a single response
    response = " ".join(relevant_chunks)
    return response

# Function to compare data across multiple PDFs
def compare_data(query):
    # Identify relevant terms to compare (this will be domain-specific)
    # For now, we assume the query is asking for a comparison across degree types or tabular data
    
    # Retrieve relevant chunks based on the query
    relevant_response = handle_query(query)
    
    # Here, you can refine the response to structure it in tabular or bullet-point format as required
    # For simplicity, we'll just return the relevant chunks
    return relevant_response

# Example function to extract data from specific pages (for the given example)
def extract_specific_data(pdf_path):
    doc = fitz.open(pdf_path)
    
    # Extract unemployment data from page 2 (example)
    page_2_text = doc.load_page(1).get_text("text")  # Page 2 corresponds to index 1
    # Implement logic to extract unemployment information (example placeholder)
    unemployment_info = "Extracted unemployment data here"
    
    # Extract tabular data from page 6 (example)
    page_6_text = doc.load_page(5).get_text("text")  # Page 6 corresponds to index 5
    # Implement logic to parse tabular data (example placeholder)
    tabular_data = "Extracted tabular data here"
    
    return unemployment_info, tabular_data

# Example usage:

# Process PDFs (replace 'pdf_file_path.pdf' with the actual file path)
pdf_file_path = "Hunter.pdf"
process_pdf(pdf_file_path)

# Handle a query (e.g., "What is the unemployment rate for bachelor's degree holders?")
query_response = handle_query("What is the unemployment rate for bachelor's degree holders?")
print("Query Response:\n",query_response)

# Compare data (e.g., comparing degree types for unemployment)
comparison_response = compare_data("Compare unemployment rates between different degrees.")
print("comparison_response:\n",comparison_response)

# Extract specific data (from page 2 and page 6)
unemployment_data, tabular_data = extract_specific_data(pdf_file_path)
print(f"Unemployment Data: {unemployment_data}")
print(f"Tabular Data: {tabular_data}")
