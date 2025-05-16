# 💻 AI-Powered Laptop Recommendation System

An intelligent laptop search and recommendation system built using **Streamlit**, **FAISS**, **Sentence Transformers**, and **LLM Chatbot (Groq API)**.

This project allows users to:
- 🔍 Perform **Semantic Search** to find laptops based on their natural language queries.
- 🎯 Get **Smart Recommendations** using filters like budget, RAM, and use-case.
- 🤖 Interact with an **AI Chatbot** for personalized laptop buying assistance.

---

## 🚀 Live App

👉 [Open the App](https://laptop-recommendation.streamlit.app/)

> Powered by Streamlit Cloud  
> Deployed from: [Laptop Recommendation GitHub Repository](https://github.com/IshaJethwa/Laptop-Recommendation)

---

## 💡 Features

- **Semantic Search:** Uses Sentence Transformers and FAISS vector search to return laptops based on your needs.
- **Filter-Based Recommendation:** Budget, RAM, and other specs-based ranking.
- **AI Assistant (LLM):** Chatbot to assist with queries, recommendations, and clarifications.
- **Streamlit Web App:** Simple and beautiful web UI built with Streamlit.

---

## 🛠 Technologies Used

- Python
- Streamlit
- FAISS (Facebook AI Similarity Search)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Groq API (`llama3-8b-8192`) for chatbot
- Pandas

---

## 📂 Project Structure

├── app.py # Streamlit main app file
├── laptops.csv # Laptop dataset
├── requirements.txt # Python dependencies
└── README.md # Project overview


---

## ⚙ Setup Instructions

1. Clone this repo:
   ```bash
   git clone https://github.com/IshaJethwa/Laptop-Recommendation.git
   cd Laptop-Recommendation
   
2. Create virtual environment (optional but recommended):
python -m venv venv
venv\Scripts\activate  # For Windows

3. Install dependencies:
streamlit run app.py

4. Run the app locally:
streamlit run app.py

🔗 Credits
Built by Isha Jethwa
LLM powered by Groq
Semantic Search using Sentence Transformers and FAISS

📢 Note
Ensure you have your Groq API key configured inside app.py where required.

