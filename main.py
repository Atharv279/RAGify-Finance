import os
import faiss
import numpy as np
import cohere
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import time

nltk.download("punkt")

# Load dataset (FinanceBench)
df = load_dataset("PatronusAI/financebench")

# Extract relevant fields
documents = [entry["evidence"][0]["evidence_text"] for entry in df["train"]]
questions = [entry["question"] for entry in df["train"]]
answers = [entry["answer"] for entry in df["train"]]

# Load embedding model
hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# API Keys (Set these in your environment variables)
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize API clients
cohere_client = cohere.Client(COHERE_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

# Function to generate embeddings
def get_embeddings(texts, model="cohere"):
    """Generate embeddings using Cohere or Hugging Face"""
    if model == "cohere":
        response = cohere_client.embed(texts=texts, model="embed-english-v2.0", input_type="text")
        return response.embeddings
    elif model == "huggingface":
        return hf_model.encode(texts, convert_to_numpy=True).tolist()

# Generate document embeddings
cohere_embeddings = get_embeddings(documents, model="cohere")
hf_embeddings = get_embeddings(documents, model="huggingface")

# Convert to NumPy for FAISS indexing
cohere_embeddings_np = np.array(cohere_embeddings, dtype="float32")
hf_embeddings_np = np.array(hf_embeddings, dtype="float32")

# Initialize FAISS indexes
d_cohere = cohere_embeddings_np.shape[1]
d_hf = hf_embeddings_np.shape[1]

cohere_index = faiss.IndexFlatL2(d_cohere)
hf_index = faiss.IndexFlatL2(d_hf)

# Add embeddings to FAISS index
cohere_index.add(cohere_embeddings_np)
hf_index.add(hf_embeddings_np)

# Retrieve Top-K relevant documents
def retrieve_documents(query, model="cohere", k=3):
    """Retrieve top K relevant documents using FAISS"""
    query_embedding = np.array(get_embeddings([query], model=model)[0], dtype="float32").reshape(1, -1)
    index = cohere_index if model == "cohere" else hf_index

    _, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0] if i < len(documents)]

# Generate answers using Gemini AI
def generate_answer_gemini(query, retrieved_docs):
    """Generate an answer using Gemini AI"""
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(f"{query}\n\n{retrieved_docs}")
    return response.text

def generate_answer(query, retrieved_docs, model="gemini"):
    if model == "gemini":
        return generate_answer_gemini(query, retrieved_docs)
    elif model == "cohere":
        return generate_answer_cohere(query, retrieved_docs)
    else:
        return "⚠️ No valid model selected!"
    
def generate_answer_cohere(query, retrieved_docs):
    try:
        co = cohere.Client("YOUR_COHERE_API_KEY")  # Replace with your actual API key
        response = co.generate(
            model="command",  # Ensure you're using the correct Cohere model
            prompt=f"Question: {query}\nContext: {retrieved_docs}\nAnswer:",
            max_tokens=100
        )
        return response.generations[0].text.strip()  # Extract and return response
    except Exception as e:
        return f"⚠️ Error generating response: {e}"
    
    
# Evaluation: Precision, Recall, F1-score, Cosine Similarity
def compute_evaluation_scores(predicted, ground_truth):
    """Calculate precision, recall, and cosine similarity between the generated and actual answers"""
    predicted_tokens = set(word_tokenize(predicted.lower()))
    ground_truth_tokens = set(word_tokenize(ground_truth.lower()))

    # Precision & Recall Calculation
    true_positives = len(predicted_tokens & ground_truth_tokens)
    false_positives = len(predicted_tokens - ground_truth_tokens)
    false_negatives = len(ground_truth_tokens - predicted_tokens)

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Compute Cosine Similarity
    predicted_embedding = np.array(get_embeddings([predicted], model="cohere")[0], dtype="float32")
    ground_truth_embedding = np.array(get_embeddings([ground_truth], model="cohere")[0], dtype="float32")
    cosine_sim = cosine_similarity([predicted_embedding], [ground_truth_embedding])[0][0]

    return {"Precision": precision, "Recall": recall, "F1-score": f1, "Cosine Similarity": cosine_sim}

# Run Cohere vs Hugging Face Comparison
queries = [
    "What are the key financial risks for banks?",
    "How do interest rates impact stock prices?",
    "Explain liquidity risk in banking."
]

results = []

for query in queries:
    for model in ["cohere", "huggingface"]:
        retrieved_docs = retrieve_documents(query, model=model)
        generated_answer = generate_answer(query, retrieved_docs)
        metrics = compute_evaluation_scores(generated_answer, answers[queries.index(query)])

        results.append({
            "Query": query,
            "Model": model,
            "Generated Answer": generated_answer,
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-score": metrics["F1-score"],
            "Cosine Similarity": metrics["Cosine Similarity"]
        })

# Save results to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("cohere_vs_huggingface_results.csv", index=False)

print(df_results)

print("✅ Script started successfully!")
