# 🚀 Cohere vs Hugging Face: Finance AI Benchmark

## 📌 Project Overview
This project compares **Cohere and Hugging Face embeddings** for **financial document retrieval and Q&A** using the **FinanceBench** dataset. The goal is to evaluate **which embedding model performs better** in extracting financial insights.

## 🎥 Walkthrough Video
📌 Check out the **video walkthrough** of the project:  
👉 https://drive.google.com/file/d/1YojZoDyrGzbmagqgOJJ678ye6imgpL7O/view?usp=sharing

## ⚡ Features
✅ FinanceBench dataset for real-world financial Q&A  
✅ **Cohere vs Hugging Face embeddings** for text retrieval  
✅ **FAISS** for fast similarity search  
✅ Answer generation using **Google Gemini & Cohere**  
✅ Performance evaluation with **Precision, Recall, F1-score, and Cosine Similarity**  

## 🛠️ Tech Stack
- **Python**
- **Cohere API**
- **Hugging Face Transformers**
- **FAISS** (for efficient document retrieval)
- **Google Gemini**
- **NLTK, Scikit-learn, Pandas**

## 📂 Dataset
We use the **FinanceBench dataset** from **PatronusAI**, which contains:
- **Financial documents** as evidence  
- **User queries** related to finance  
- **Ground truth answers** for evaluation  

## 🔬 Model Comparison
We evaluate **Cohere embeddings** and **Hugging Face's all-MiniLM-L6-v2 model** based on:
- **Retrieval performance** (using FAISS)  
- **Generated answer quality** (using Gemini & Cohere)  
- **Evaluation Metrics**: Precision, Recall, F1-score, Cosine Similarity  

## 📊 Results
Results are stored in **cohere_vs_huggingface_results.csv**  
- Cohere performed **better in finance-specific tasks**  
- Hugging Face had **better recall**  
- Overall, **Cohere embeddings had higher cosine similarity**  

## 🚀 How to Run
1️⃣ Clone this repository:  
```bash
git clone https://github.com/yourusername/finance-ai-benchmark
cd finance-ai-benchmark


2️⃣ Install dependencies:
pip install -r requirements.txt

3️⃣ Set up API keys in environment variables:
export COHERE_API_KEY="your_cohere_api_key"
export GOOGLE_API_KEY="your_google_api_key"

4️⃣ Run the main script:
python main.py

5️⃣ View results in cohere_vs_huggingface_results.csv

🎯 Future Improvements
Add Voyage AI once accessible without a credit card
Test Athina AI if sandbox version is released
Expand to other finance-specific NLP models
📢 Connect With Me
🔹 LinkedIn: www.linkedin.com/in/atharv-patil-bab53a284
🔹 GitHub: https://github.com/Atharv279

🤝 I'm open to AI/NLP roles! Let's connect!

#AI #MachineLearning #FinanceAI #Cohere #HuggingFace #NLP #OpenToWork