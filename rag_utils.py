import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.settings import Settings
import faiss

# Constants
UPLOAD_DIR = "./pdfs"

# Load and index PDFs
def load_and_index_pdfs():
    documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.embed_model = embed_model

    faiss_index = faiss.IndexFlatL2(768)  # 768 for BAAI/bge-base-en-v1.5
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    llm = get_llm()
    Settings.llm = llm
    return index, llm


# Initialize GROQ LLM
def get_llm():
    return Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))


# Rewrite the user query using the LLM
def rewrite_query(original_query: str, llm) -> str:
    prompt = (
        "Rephrase the following user question to be more specific and clear "
        "for use in document retrieval and semantic search:\n\n"
        f"User Question: {original_query}\n\nRewritten Question:"
    )
    response = llm.complete(prompt)
    return response.text.strip()


# Create a query engine
def create_query_engine(index, llm):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=6)
    reranker = SentenceTransformerRerank(top_n=3, model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return RetrieverQueryEngine(retriever=retriever, node_postprocessors=[reranker])
