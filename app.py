# app.py
import streamlit as st
from rag_utils import load_and_index_pdfs, create_query_engine, rewrite_query

st.set_page_config(page_title="📄 GROQ PDF Q&A", layout="centered")
st.title("📄 Ask Questions from Your PDFs")

index, llm = load_and_index_pdfs()
query_engine = create_query_engine(index, llm)

query = st.text_input("Ask a question from your PDFs:")

if query:
    with st.spinner("Thinking..."):
        response = query_engine.query(query)

    
    st.markdown("### Answer:")
    st.write(response.response)  # just clean answer text
