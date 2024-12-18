import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from docx import Document
import time
import os

# Function to extract QA pairs from .docx
def extract_qa_pairs_from_docx(file_path):
    """Extracts QA pairs from a .docx file."""
    doc = Document(file_path)
    content = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    qa_pairs = []

    question = None
    answer = ""
    for line in content:
        if line.endswith("?"):  # Questions end with '?'
            if question is not None:
                qa_pairs.append({"question": question, "answer": answer.strip()})
            question = line
            answer = ""  # Reset for the next answer
        else:
            answer += " " + line
    if question:  # Add the last QA pair
        qa_pairs.append({"question": question, "answer": answer.strip()})
    return qa_pairs

# Load models
@st.cache_resource
def load_retrieval_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def load_generation_model():
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
    return tokenizer, model

@st.cache_resource
def load_knowledge_base(file_path):
    """Load and index the knowledge base."""
    knowledge_base = extract_qa_pairs_from_docx(file_path)
    texts = [entry['question'] + ' ' + entry['answer'] for entry in knowledge_base]

    # Create FAISS index
    retrieval_model = load_retrieval_model()
    embeddings = retrieval_model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, texts

# Define functions
def retrieve_context(query, index, texts, retrieval_model, top_k=3):
    query_embedding = retrieval_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [texts[idx] for idx in indices[0]]

def generate_response(query, context_list, tokenizer, model):
    context = "\n".join(context_list)
    input_text = f"""
    Context: {context}

    You are a medical chatbot specializing in health-related questions. Answer this query:
    {query}
    """
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=1500, num_beams=5)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app
def main():
    st.title("Medical Chatbot")
    st.write("Ask any health-related questions, and I'll provide an answer.")

    # Load models and knowledge base
    retrieval_model = load_retrieval_model()
    tokenizer, generation_model = load_generation_model()

    # Google Drive file path
    file_path = "QA_data_enlarged.docx"  # Update to match your Google Drive path
    if not os.path.exists(file_path):
        st.error("Knowledge base file not found. Please check the file path.")
        return

    st.info("Loading knowledge base...")
    index, texts = load_knowledge_base(file_path)
    st.success("Knowledge base loaded successfully!")

    # User query
    query = st.text_input("Enter your query:")
    if query:
        start_time = time.time()
        retrieved_context = retrieve_context(query, index, texts, retrieval_model)
        response = generate_response(query, retrieved_context, tokenizer, generation_model)
        response_time = time.time() - start_time

        # Display results
        st.subheader("Response")
        st.write(response)
        st.subheader("Retrieved Context")
        st.write(retrieved_context)
        st.write(f"Response generated in {response_time:.2f} seconds")

if __name__ == "__main__":
    main()
