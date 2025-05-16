import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# -------------------- Load data and models --------------------
df = pd.read_csv("laptops.csv")
df['ram_clean'] = df['ram'].str.extract('(\d+)').astype(float)
df['ram_gb'] = df['ram_clean'].astype(int)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
laptop_embeddings = embed_model.encode(df['name'].tolist(), normalize_embeddings=True)
dimension = laptop_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(laptop_embeddings)

client = Groq(api_key="YOUR_GROQ_API_KEY")  # Replace with your key

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="AI Laptop Finder", page_icon="💻", layout="wide")



# -------------------- Main Title --------------------
st.title("💻 AI-Powered Laptop Search & Recommendations")
st.markdown("> Find your perfect laptop by budget, specs, or use case — powered by Semantic Search & AI Chatbot.")

# -------------------- Sidebar --------------------
st.sidebar.title("⚙️ Filters & Chat")
st.sidebar.markdown("Use the filters or directly talk to the chatbot below.")

budget = st.sidebar.slider("Set Max Budget (₹)", int(df['price'].min()), int(df['price'].max()), 50000)
ram_options = sorted(df['ram_clean'].unique())
selected_ram = st.sidebar.selectbox("Minimum RAM (Optional)", ["Any"] + ram_options)

st.sidebar.markdown("### 💬 Quick Chat")
sidebar_query = st.sidebar.text_input("Ask your laptop question")

if sidebar_query:
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful laptop recommendation assistant."},
                {"role": "user", "content": sidebar_query}
            ],
            temperature=0.5,
            max_tokens=500
        )
        st.sidebar.success(response.choices[0].message.content)
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# ✅ Filter Preview
with st.sidebar.expander("🔎 Filtered Laptops Preview", expanded=True):
    preview_df = df[df['price'] <= budget]
    if selected_ram != "Any":
        preview_df = preview_df[preview_df['ram_clean'] >= float(selected_ram)]
    st.write(preview_df[['name', 'price', 'ram']].head(5))

# -------------------- Tabs --------------------
tabs = st.tabs(["🔍 Semantic Search", "🎯 Recommendations", "🤖 LapGuru"])

# -------------------- Semantic Search Tab --------------------
with tabs[0]:
    st.header("🤖 Meet LapGuru - Your AI Laptop Expert")
              
    user_query = st.text_input("Describe what you need", placeholder="e.g., i7 laptop for office work under 70000")

    if user_query:
        query_embedding = embed_model.encode([user_query], normalize_embeddings=True)
        distances, indices = index.search(query_embedding, 5)

        st.markdown("### Top Results")
        for i in indices[0]:
            laptop = df.iloc[i]
            st.markdown(f"""
                <div style='border: 1px solid #ddd; border-radius:12px; padding:15px; margin-bottom:15px; background-color:#f9f9f9; color:#000;'>
                    <h4>{laptop['name']}</h4>
                    <p><b>Processor:</b> {laptop['processor']} | <b>RAM:</b> {laptop['ram']} | <b>Price:</b> ₹{laptop['price']}</p>
                </div>
            """, unsafe_allow_html=True)

# -------------------- Recommendations Tab --------------------
with tabs[1]:
    st.header("🎯 Smart Laptop Recommendations")

    filtered_df = df[df['price'] <= budget]
    if selected_ram != "Any":
        filtered_df = filtered_df[filtered_df['ram_clean'] >= float(selected_ram)]

    filtered_df = filtered_df.sort_values(['ram_gb', 'price'], ascending=[False, True])

    if filtered_df.empty:
        st.warning("⚠ No laptops found matching your filters.")
    else:
        st.success(f"Showing Top {min(5, len(filtered_df))} Recommendations under ₹{budget}")
        cols = st.columns(2)
        for i, (_, row) in enumerate(filtered_df.head(5).iterrows()):
            with cols[i % 2]:
                st.markdown(f"""
                    <div style='border: 2px solid #34A853; border-radius:12px; padding:15px; margin-bottom:15px; background-color:#e8f5e9; color:#000; font-weight:500; font-size:16px;'>
                        <h4>{row['name']}</h4>
                        <p><b>Processor:</b> {row['processor']}<br>
                        <b>RAM:</b> {row['ram']}<br>
                        <b>Price:</b> ₹{row['price']}</p>
                        <span style='background-color:#34A853; color:white; padding:5px 8px; border-radius:5px;'>Best Pick</span>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("🤖 Need More Guidance? AI Assistant")
    assist_query = st.text_input("Ask your custom question (e.g., best gaming laptop under 60000)")

    if assist_query:
        try:
            assist_response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful laptop recommendation assistant."},
                    {"role": "user", "content": assist_query}
                ],
                temperature=0.5,
                max_tokens=500
            )
            st.success("AI Suggests:")
            st.write(assist_response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error: {e}")

# -------------------- Chatbot Only Tab --------------------
with tabs[2]:
    st.header("🤖 Your Personal Chatbot")
    chat_query = st.text_input("Chat with AI")

    if chat_query:
        try:
            chat_response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful laptop recommendation assistant."},
                    {"role": "user", "content": chat_query}
                ],
                temperature=0.5,
                max_tokens=500
            )
            st.success(chat_response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error: {e}")
