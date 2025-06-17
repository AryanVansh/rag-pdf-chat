# app.py
import streamlit as st
from rag_utils import load_and_index_pdfs, create_query_engine, rewrite_query

st.set_page_config(page_title="📄 GROQ PDF Q&A", layout="centered")
st.title("📄 Ask Questions from Your PDFs")

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(f"./pdfs/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.read())

    index, llm = load_and_index_pdfs()
    query_engine = create_query_engine(index, llm)

    user_input = st.text_input("Ask a question")

    if user_input:
        rewritten = rewrite_query(user_input, llm)
        st.write("**Rewritten Query:**", rewritten)

        with st.spinner("Thinking..."):
            response = query_engine.query(rewritten)
        st.write("**Answer:**", response)
