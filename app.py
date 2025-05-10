import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load laptop dataset
final_df = pd.read_csv("laptops.csv")

# Load model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def create_index():
    descriptions = final_df['description'].astype(str).tolist()
    embeddings = model.encode(descriptions, show_progress_bar=True)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

index, embeddings = create_index()

# Semantic search
def semantic_search(query, top_k=10):
    query_embedding = model.encode([query])
    scores, indices = index.search(np.array(query_embedding), top_k)
    results = final_df.iloc[indices[0]]
    return results

# Streamlit UI
st.title("💻 AI-Powered Laptop Search")
st.markdown("Ask me for laptops by budget, processor, RAM, or usage (e.g., gaming, office, editing).")

query = st.text_input("What are you looking for?", placeholder="e.g., i7 laptop for video editing under 60000")

if query:
    with st.spinner("Searching..."):
        results = semantic_search(query)
        st.success("Here are some laptops that match your needs:")
        st.write(results[['brand', 'name', 'price', 'processor', 'ram', 'description']])
