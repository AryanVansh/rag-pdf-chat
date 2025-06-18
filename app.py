# app.py

import os
import streamlit as st
from rag_utils import load_and_index_pdfs, create_query_engine, rewrite_query

st.set_page_config(page_title="📄 GROQ PDF Q&A", layout="centered")
st.title("📄 Ask Questions from Your PDFs")

UPLOAD_DIR = "./pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files and "query_engine" not in st.session_state:
    for file in uploaded_files:
        with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
            f.write(file.read())
    st.success(f"{len(uploaded_files)} file(s) uploaded.")

    index, llm = load_and_index_pdfs()
    query_engine = create_query_engine(index, llm, use_rerank=True)
    st.session_state.query_engine = query_engine
    st.session_state.llm = llm
else:
    query_engine = st.session_state.get("query_engine")

if query_engine:
    user_query = st.text_input("Ask a question from your PDFs:")

    if user_query:
        with st.spinner("Answering..."):
            # Optional rewrite (can comment for speed)
            # rewritten_query = rewrite_query(user_query, st.session_state.llm)
            response = query_engine.query(user_query)
        st.markdown("### Answer:")
        st.success(response.response)
