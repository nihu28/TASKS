import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Load a pre-trained sentence transformer model to generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to scrape content from a webpage
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')  # Assuming the content is in <p> tags
    text_content = [p.get_text() for p in paragraphs]
    return text_content

# Convert content to vector embeddings
def text_to_embeddings(text_content):
    embeddings = model.encode(text_content)
    return embeddings

# Function to create FAISS index for the embeddings
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]  # Dimensionality of the embeddings
    index = faiss.IndexFlatL2(dim)  # Using L2 distance for similarity search
    index.add(embeddings)
    return index

# Example URL (You can change this to any valid URL)
url = "https://www.uchicago.edu/"
content = scrape_website(url)
embeddings = text_to_embeddings(content)
index = create_faiss_index(embeddings)

def query_to_embedding(query):
    return model.encode([query])

def search_similar_content(query_embedding, index, content, top_k=5):
    # Ensure query_embedding is a 2D array with one row
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Search for the most similar chunks to the query
    D, I = index.search(query_embedding, top_k)
    results = [content[i] for i in I[0]]
    return results

# Example query
query = "What is the university's mission?"
query_embedding = query_to_embedding(query)
top_content = search_similar_content(query_embedding, index, content)

# Display the top retrieved chunks
for idx, chunk in enumerate(top_content):
    print(f"Result {idx+1}: {chunk}\n")

# Initialize a text generation pipeline using a pre-trained model
generator = pipeline('text-generation', model='gpt2')  # You can replace this with GPT-3 or GPT-J for better results

# Create a prompt that includes the retrieved content
def generate_response(query, top_content):
    context = "\n".join(top_content)  # Join the top chunks of content as context
    prompt = f"Based on the following information, answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    
    # Set max_new_tokens instead of max_length to allow generation of a specific number of new tokens
    response = generator(prompt, max_new_tokens=100, num_return_sequences=1)  # Adjust 100 based on your needs
    return response[0]['generated_text']

# Example usage
response = generate_response(query, top_content)
print(f"Generated Response: {response}")
