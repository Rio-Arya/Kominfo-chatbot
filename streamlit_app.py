import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import huggingface_hub
import torch

# Pinecone initialization
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    st.error("Pinecone API key not found. Please set the PINECONE_API_KEY environment variable.")
    st.stop()

pc = Pinecone(api_key=api_key)

# Pinecone index configuration
index_name = 'prototype'
dimension = 768  # Adjust based on your embeddings

# Connect to Pinecone index
try:
    index = pc.Index(index_name)
    st.success(f"Connected to Pinecone index: {index_name}")
except Exception as e:
    st.error(f"Failed to connect to Pinecone index: {e}")
    st.stop()

# Hugging Face login
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Sentence-BERT model
try:
    model_sentence_bert = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2', device=device)
    st.success("Sentence-BERT model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load Sentence-BERT model: {e}")
    st.stop()

# Function to get query embedding
def get_query_embedding(query):
    return model_sentence_bert.encode([query])

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = get_query_embedding(query)
    query_embedding = np.array(query_embedding).astype('float32')

    try:
        results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
        print("Pinecone query successful:", results)
    except Exception as e:
        print("Pinecone query failed:", e)
        return []

    relevant_chunks = []
    for result in results['matches']:
        metadata = result.get('metadata', {})
        content = metadata.get('content', "Konten tidak tersedia.")
        source = metadata.get('source', "Sumber tidak diketahui.")
        relevant_chunks.append({"content": content, "source": source})

    return relevant_chunks[:top_k]

# Load Gemma-2-2B model and tokenizer
model_name = "google/gemma-1.1-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=True
    ).to(device)
    st.success("Gemma-1.1-2b-it model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load Gemma-1.1-2b-it model: {e}")
    st.stop()

# Main function to handle questions
def handle_question(query):
    relevant_chunks = retrieve_relevant_chunks(query)

    context = " ".join([chunk['content'] for chunk in relevant_chunks])[:1000]
    input_text = f"Konteks: {context}\n\nPertanyaan: {query}\nJawaban: "

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=100,
        temperature=0.7,
        top_k=3,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if relevant_chunks:
        sources = f"Sumber informasi: {relevant_chunks[0]['source']}"
    else:
        sources = "Sumber informasi tidak ditemukan."

    return response, sources

# Streamlit app
st.title("RAG Chatbot")

query = st.text_input("Masukkan pertanyaan Anda:")

if st.button("Cari Jawaban"):
    if query:
        response, sources = handle_question(query)
        st.write(f"**Respon:** {response}")
        st.write(f"**{sources}**")
    else:
        st.write("Silakan masukkan pertanyaan terlebih dahulu.")