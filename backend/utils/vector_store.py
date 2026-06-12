import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """
    Cached embedding model loader. Avoids reloading the model on every Streamlit rerun.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

def create_vector_store(documents, embeddings):
    """
    Creates a FAISS vector store from LangChain Document objects.
    """
    vector_db = FAISS.from_documents(documents, embeddings)
    return vector_db

def load_local_vector_store(db_path, embeddings):
    """
    Loads a FAISS vector store from the local filesystem.
    """
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return None

def save_local_vector_store(vector_db, db_path):
    """
    Saves a FAISS vector store to the local filesystem.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    vector_db.save_local(db_path)