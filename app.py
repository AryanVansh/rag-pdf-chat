# app.py
import os
import streamlit as st
from rag_utils import load_and_index_pdfs, create_query_engine

st.set_page_config(page_title="📄 GROQ PDF Q&A", layout="centered")
st.title("📄 Ask Questions from Your PDFs")

UPLOAD_DIR = "pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Upload PDFs
uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Save uploaded PDFs
    for file in uploaded_files:
        with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
            f.write(file.read())
    st.success(f"{len(uploaded_files)} file(s) uploaded.")

    # Load index and setup query engine
    index, llm = load_and_index_pdfs()
    query_engine = create_query_engine(index, llm)

    # Input question
    query = st.text_input("Ask a question from your PDFs:")
    if query:
        with st.spinner("Searching..."):
            response = query_engine.query(query)
            st.markdown("### Answer:")
            st.write(response.response)  # Only show actual answer
