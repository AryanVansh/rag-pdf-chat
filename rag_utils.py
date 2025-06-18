import os
import torch
import faiss
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.settings import Settings

# Load environment variables
load_dotenv()

# Auto device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
UPLOAD_DIR = "./pdfs"

# Load and index PDFs
def load_and_index_pdfs():
    # Load documents from ./pdfs
    documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()

    # Set embedding model with GPU/CPU device
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
        device=device
    )
    Settings.embed_model = embed_model

    # Set up FAISS vector store
    faiss_index = faiss.IndexFlatL2(768)  # Vector size for BAAI/bge-base-en-v1.5
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    # Build index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # Load GROQ LLM
    llm = get_llm()
    Settings.llm = llm

    return index, llm

# Initialize GROQ LLM
def get_llm():
    return Groq(
        model="llama3-70b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

# Rewrite user query using LLM (optional, for clarity enhancement)
def rewrite_query(original_query: str, llm) -> str:
    prompt = (
        "Rephrase the following user question to be more specific and clear "
        "for use in document retrieval and semantic search:\n\n"
        f"User Question: {original_query}\n\nRewritten Question:"
    )
    response = llm.complete(prompt)
    return response.text.strip()

# Create the full query engine with retriever + reranker + llm
def create_query_engine(index, llm):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=6)

    reranker = SentenceTransformerRerank(
        top_n=3,
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=device
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
        llm=llm
    )
    return query_engine
