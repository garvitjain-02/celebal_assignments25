import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load data
df = pd.read_csv("Trainingdataset.csv") 
df.dropna(inplace=True)

# Convert rows to text for embedding
def row_to_text(row):
    try:
        return (
            f"{row['Gender']} applicant with {row['Education']} education, "
            f"Credit History: {row['Credit_History']}, "
            f"Loan Amount: {row['LoanAmount']}, "
            f"Income: {row['ApplicantIncome']} => Loan "
            f"{'approved' if row['Loan_Status'] == 'Y' else 'rejected'}."
        )
    except KeyError as e:
        return "Incomplete row"

documents = df.apply(row_to_text, axis=1).tolist()

# Embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(documents, convert_to_numpy=True)
embedding_dim = embeddings.shape[1]

# Create and index FAISS
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Gemini API setup
genai.configure(api_key="AIzaSyDU0p_RCD870oxsu6vmClvG4kqK_ASGzZc")
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# UI
st.title("Loan Approval RAG Chatbot (Gemini API)")
query = st.text_input("Ask a loan-related question:")

if query:
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k=5)
    top_docs = [documents[i] for i in I[0]]
    
    prompt = f"""Context:
{chr(10).join(top_docs)}

Question: {query}
Answer:"""

    try:
        response = gemini_model.generate_content(prompt)
        st.write("### Answer")
        st.write(response.text)
    except Exception as e:
        st.error(f"Error from Gemini: {e}")
