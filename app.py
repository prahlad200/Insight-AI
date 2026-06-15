import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import sqlite3
import os
from datetime import datetime
import concurrent.futures
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
try:
    from openai import OpenAI
except ImportError:
    pass

# --- 1. MODERN THEME & INTERACTIVE UI CONFIGURATION ---
st.set_page_config(page_title="InsightAI | Tech Sentiment Platform", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* 1. Flat, stark metric cards like code windows */
    div[data-testid="metric-container"] {
        background-color: #1A1A1A; 
        border: 1px solid #333333;
        padding: 20px; 
        border-radius: 8px; 
        box-shadow: none;
        transition: border-color 0.3s ease;
    }
    /* 2. Hover effect mimicking Carbon's yellow logo */
    div[data-testid="metric-container"]:hover { 
        border-color: #F8DF72; 
    }
    /* 3. Sleek, thin borders for input fields */
    .stTextInput>div>div>input {
        background-color: #1A1A1A;
        border: 1px solid #333333;
        color: #FFFFFF;
    }
    /* 4. Sharp, terminal-style buttons */
    .stButton>button { 
        width: 100%; 
        border-radius: 6px; 
        font-weight: 600; 
        background-color: #000000;
        border: 1px solid #333333;
        transition: all 0.2s ease; 
    }
    .stButton>button:hover {
        border-color: #F8DF72;
        color: #F8DF72;
        background-color: #1A1A1A;
    }
    /* 5. Customizing the metric numbers */
    [data-testid="stMetricSimpleValue"] { 
        font-size: 2.5rem; 
        font-weight: 400; 
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATABASE & VECTOR PERSISTENCE LAYER ---
DB_FILE = "insights.db"
FAISS_INDEX_FILE = "insights.index"
MAP_FILE = "insights_map.txt"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, search_query TEXT,
            headline TEXT, headline_vibe TEXT, comment_text TEXT, comment_vibe TEXT,
            comment_emotion TEXT, confidence REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, password TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, 
            role TEXT, content TEXT, timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def build_and_save_vector_index():
    """Reads all database records, encodes them, and persists the FAISS index to disk."""
    conn = sqlite3.connect(DB_FILE)
    db_data = pd.read_sql_query("SELECT id, search_query, headline, comment_text FROM ai_insights ORDER BY id DESC", conn)
    conn.close()
    
    if db_data.empty:
        if os.path.exists(FAISS_INDEX_FILE): os.remove(FAISS_INDEX_FILE)
        if os.path.exists(MAP_FILE): os.remove(MAP_FILE)
        return

    documents = [f"[Source Record ID: {r['id']}] Target Topic: {r['search_query']} | Article Title: {r['headline']} | User Feedback: {r['comment_text']}" for _, r in db_data.iterrows()]
    doc_embeddings = vector_embedder.encode(documents, convert_to_numpy=True)
    
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(MAP_FILE, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.replace("\n", " ") + "\n")

def save_to_database(df_records, query_used):
    conn = sqlite3.connect(DB_FILE)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for _, row in df_records.iterrows():
        conn.execute("""
            INSERT INTO ai_insights (timestamp, search_query, headline, headline_vibe, comment_text, comment_vibe, comment_emotion, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (current_time, query_used if query_used.strip() else "Global Top Stories", row["Headline"], row["Headline Vibe"], row["Top Community Comment"], row["Comment Vibe"], row["Comment Emotion"], row["Raw_Confidence"]))
    conn.commit()
    conn.close()
    build_and_save_vector_index()

def register_user(username, password):
    try:
        conn = sqlite3.connect(DB_FILE)
        clean_username = username.strip().lower()
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (clean_username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    clean_username = username.strip().lower()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (clean_username, password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

def reset_password(username, new_password):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    clean_username = username.strip().lower()
    
    cursor.execute("SELECT * FROM users WHERE username = ?", (clean_username,))
    if cursor.fetchone() is None:
        conn.close()
        return False
        
    cursor.execute("UPDATE users SET password = ? WHERE username = ?", (new_password, clean_username))
    conn.commit()
    conn.close()
    return True

init_db()

# --- 3. SESSION STATE INITIALIZATION ---
if "active_data" not in st.session_state: st.session_state.active_data = None
if "current_query" not in st.session_state: st.session_state.current_query = "None (Awaiting Execution)"
if "logged_in_user" not in st.session_state: st.session_state.logged_in_user = None
if "llm_provider" not in st.session_state: st.session_state.llm_provider = "Google Gemini"
if "llm_model" not in st.session_state: st.session_state.llm_model = "gemini-2.5-flash"
if "llm_api_key" not in st.session_state: st.session_state.llm_api_key = ""

# --- 4. USER GATEKEEPER LAYER (LOGIN) ---
if st.session_state.logged_in_user is None:
    st.title("🔒 InsightAI Secure Gateway")
    st.write("Please authenticate to access the enterprise intelligence platform.")
    
    tab_login, tab_signup, tab_reset = st.tabs(["Sign In", "Create Account", "Forgot Password"])
    
    with tab_login:
        user_input = st.text_input("Username", key="login_user")
        pass_input = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log In", type="primary"):
            if verify_user(user_input, pass_input):
                st.session_state.logged_in_user = user_input.strip().lower()
                st.success(f"Welcome back, {st.session_state.logged_in_user}!")
                st.rerun()
            else:
                st.error("Invalid credentials.")
                
    with tab_signup:
        new_user = st.text_input("Choose Username", key="signup_user")
        new_pass = st.text_input("Choose Password", type="password", key="signup_pass")
        if st.button("Register", type="primary"):
            if new_user and new_pass:
                if register_user(new_user, new_pass):
                    st.success("Account created successfully! Please switch to the 'Sign In' tab.")
                else:
                    st.error("Username already exists.")
                    
    with tab_reset:
        st.info("In a production environment, this would require an email verification token.")
        reset_user = st.text_input("Account Username", key="reset_user")
        reset_pass = st.text_input("New Password", type="password", key="reset_pass")
        if st.button("Update Password", type="primary"):
            if reset_user and reset_pass:
                if reset_password(reset_user, reset_pass):
                    st.success("Password reset successfully! Please switch to the 'Sign In' tab.")
                else:
                    st.error("Username not found in the system.")
                    
    st.stop()

# --- 5. LOAD PRE-TRAINED AI MODELS (CACHED) ---
@st.cache_resource
def load_sentiment_brain():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")

@st.cache_resource
def load_emotion_brain():
    return pipeline("sentiment-analysis", model="bhadresh-savani/distilbert-base-uncased-emotion")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

with st.spinner("Optimizing AI computational graph..."):
    sentiment_ai = load_sentiment_brain()
    emotion_ai = load_emotion_brain()
    vector_embedder = load_embedding_model()

def clean_html(raw_html):
    if not raw_html: return ""
    return re.sub('<.*?>', '', raw_html)[:300]

def fetch_single_story_and_comment(story_id):
    try:
        item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        item_data = requests.get(item_url, timeout=5).json()
        headline_text = item_data.get('title', '')
        kids = item_data.get('kids', [])
        comment_text = "No comments yet"
        if kids:
            comment_url = f"https://hacker-news.firebaseio.com/v0/item/{kids[0]}.json"
            comment_data = requests.get(comment_url, timeout=5).json()
            comment_text = clean_html(comment_data.get('text', ''))
        return {"headline": headline_text, "comment": comment_text if comment_text else "Empty comment text"}
    except Exception:
        return None

# --- 6. SIDEBAR NAVIGATION & CONTROLS ---
with st.sidebar:
    st.title("🧠 InsightAI")
    st.write(f"Active Session: **{st.session_state.logged_in_user}**")
    
    if st.button("Log Out"):
        st.session_state.logged_in_user = None
        st.session_state.llm_api_key = "" 
        st.rerun()
        
    st.markdown("---")
    
    # --- DYNAMIC MULTI-LLM PROVIDER CONFIGURATOR ---
    st.markdown("### 🔑 AI Framework Configuration")
    
    chosen_provider = st.selectbox(
        "Framework Engine:",
        ["Google Gemini", "OpenAI"],
        index=0 if st.session_state.llm_provider == "Google Gemini" else 1
    )
    st.session_state.llm_provider = chosen_provider
    
    if chosen_provider == "Google Gemini":
        model_options = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"]
    else:
        model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        
    try:
        current_model_idx = model_options.index(st.session_state.llm_model)
    except ValueError:
        current_model_idx = 0
        
    chosen_model = st.selectbox("Model Architecture:", model_options, index=current_model_idx)
    st.session_state.llm_model = chosen_model
    
    secret_key_name = "GEMINI_API_KEY" if chosen_provider == "Google Gemini" else "OPENAI_API_KEY"
    try:
        st.session_state.llm_api_key = st.secrets[secret_key_name]
        st.success("API Connection: Vault Active")
    except Exception:
        saved_key = st.text_input(
            f"{chosen_provider} API Key:", 
            value=st.session_state.llm_api_key, 
            type="password"
        )
        if saved_key != st.session_state.llm_api_key:
            st.session_state.llm_api_key = saved_key
        
    st.markdown("---")
    
    app_mode = st.radio(
        "Platform Navigation",
        ["📡 Live Dashboard", "📝 Raw Data Feed", "🗄️ Historical Warehouse", "💬 AI Data Analyst"]
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ Pipeline Configurator")
    search_query = st.text_input("Topic Target:", placeholder="e.g., OpenAI, Nvidia")
    num_posts = st.slider("Sample Size Volume:", 3, 15, 5)
    
    if st.button("Execute High-Speed Run", type="primary"):
        parsed_results = []
        used_fallback = False
        raw_stories_data = []
        
        try:
            if search_query.strip():
                url = f"https://hn.algolia.com/api/v1/search_by_date?query={search_query}&tags=story&hitsPerPage={num_posts}"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    hits = response.json().get('hits', [])
                    story_ids = [hit.get('objectID') for hit in hits if hit.get('objectID')]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        results = executor.map(fetch_single_story_and_comment, story_ids)
                        raw_stories_data = [r for r in results if r is not None]
                else: used_fallback = True
            else:
                url = "https://hacker-news.firebaseio.com/v0/topstories.json"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    story_ids = response.json()[:num_posts]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        results = executor.map(fetch_single_story_and_comment, story_ids)
                        raw_stories_data = [r for r in results if r is not None]
                else: used_fallback = True
                    
            if raw_stories_data and not used_fallback:
                for data in raw_stories_data:
                    h_text, c_text = data["headline"], data["comment"]
                    c_vibe, c_emo = "Neutral", "Neutral"
                    if h_text:
                        s_output = sentiment_ai(h_text[:500])[0]
                        sentiment_label = s_output['label'].capitalize()
                        confidence_score = round(s_output['score'] * 100, 1)
                        if c_text and c_text != "No comments yet":
                            c_vibe = sentiment_ai(c_text[:500])[0]['label'].capitalize()
                            c_emo = emotion_ai(c_text[:500])[0]['label'].capitalize()
                        parsed_results.append({
                            "Headline": h_text, "Headline Vibe": sentiment_label,
                            "Top Community Comment": c_text, "Comment Vibe": c_vibe,
                            "Comment Emotion": c_emo, "Raw_Confidence": confidence_score,
                            "Confidence": f"{confidence_score}%"
                        })
            else: used_fallback = True
        except Exception: used_fallback = True

        if used_fallback:
            st.error("Network failed. Skipping fallback to maintain data integrity.")
                
        if parsed_results:
            st.session_state.active_data = pd.DataFrame(parsed_results)
            st.session_state.current_query = search_query if search_query else "Global Trending"
            save_to_database(st.session_state.active_data, st.session_state.current_query)
            st.success("Pipeline Execution Complete!")

# --- 7. MAIN WORKSPACE ROUTING ---

if app_mode == "📡 Live Dashboard":
    st.header(f"Active Intelligence Stream: {st.session_state.current_query}")
    if st.session_state.active_data is not None:
        raw_df = st.session_state.active_data
        
        st.markdown("### 🔍 Interactive Stream Filters")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            vibe_opts = ["All Vibes"] + list(raw_df['Comment Vibe'].unique())
            selected_vibe = st.selectbox("Filter by Sentiment Vibe:", vibe_opts)
        with f_col2:
            emotion_opts = ["All Emotions"] + list(raw_df['Comment Emotion'].unique())
            selected_emotion = st.selectbox("Filter by Emotion Matrix:", emotion_opts)
            
        df = raw_df.copy()
        if selected_vibe != "All Vibes":
            df = df[df['Comment Vibe'] == selected_vibe]
        if selected_emotion != "All Emotions":
            df = df[df['Comment Emotion'] == selected_emotion]
            
        if df.empty:
            st.warning("No records match the active filter configuration. Reset filters to view analytics.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Ingested Nodes (Filtered)", len(df))
            col2.metric("Primary Emotion", df['Comment Emotion'].mode()[0] if not df['Comment Emotion'].empty else "N/A")
            col3.metric("Avg Confidence", f"{round(df['Raw_Confidence'].mean(), 1)}%")
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            
            carbon_palette = ['#82AAFF', '#C792EA', '#F07178', '#C3E88D', '#FFCB6B']
            
            with c1:
                st.subheader("Sentiment Distribution")
                st.caption("Shows macro-level community consensus by categorizing unstructured text into Positive, Neutral, or Negative buckets.")
                sentiment_counts = df['Comment Vibe'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Frequency']
                fig_sent = px.bar(
                    sentiment_counts, x='Sentiment', y='Frequency', color='Sentiment', 
                    template='plotly_dark',
                    color_discrete_sequence=carbon_palette
                )
                fig_sent.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20), height=300)
                st.plotly_chart(fig_sent, use_container_width=True)
                
            with c2:
                st.subheader("Emotion Matrix")
                st.caption("Provides semantic depth by isolating the exact psychological drivers (e.g., Fear, Joy) behind the community's reaction.")
                emotion_counts = df['Comment Emotion'].value_counts().reset_index()
                emotion_counts.columns = ['Emotion', 'Frequency']
                fig_emo = px.bar(
                    emotion_counts, x='Emotion', y='Frequency', color='Emotion', 
                    template='plotly_dark',
                    color_discrete_sequence=carbon_palette
                )
                fig_emo.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20), height=300)
                st.plotly_chart(fig_emo, use_container_width=True)
                
            st.markdown("---")
            st.subheader("📈 Predictive Sentiment Trend Analysis")
            st.caption("Projects future sentiment trajectory by calculating the average rate of variance change across the chronological data sequence.")
            
            vibe_map = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
            numerical_vibes = df['Comment Vibe'].map(vibe_map).tolist()
            
            if len(numerical_vibes) >= 2:
                steps = list(range(1, len(numerical_vibes) + 1))
                
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=steps, y=numerical_vibes, mode='lines+markers', name='Observed Vibe Index', line=dict(color='#82AAFF')))
                
                last_val = numerical_vibes[-1]
                avg_delta = np.mean(np.diff(numerical_vibes)) if len(numerical_vibes) > 1 else 0
                
                future_steps = [len(numerical_vibes) + 1, len(numerical_vibes) + 2, len(numerical_vibes) + 3]
                future_vals = [np.clip(last_val + (avg_delta * i), -1.0, 1.0) for i in range(1, 4)]
                
                proj_steps = [steps[-1]] + future_steps
                proj_vals = [numerical_vibes[-1]] + future_vals
                
                fig_trend.add_trace(go.Scatter(x=proj_steps, y=proj_vals, mode='lines+markers', line=dict(dash='dash', color='#FFCB6B'), name='Predictive Forecast Trajectory'))
                fig_trend.update_layout(
                    template='plotly_dark',
                    yaxis=dict(tickmode='array', tickvals=[-1, 0, 1], ticktext=['Negative', 'Neutral', 'Positive']),
                    xaxis_title="Chronological Queue Order (Filtered Ingested Sequence)",
                    margin=dict(l=20, r=20, t=20, b=20), height=350
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Additional filtered sample volume size is required to compile forecasting predictions.")
                
            st.markdown("---")
            st.subheader("Multivariate Cross-Correlation")
            st.caption("Statistical validation matrix checking for conceptual alignment between the independent Sentiment and Emotion AI models.")
            st.dataframe(pd.crosstab(df['Comment Vibe'], df['Comment Emotion']), use_container_width=True)
    else:
        st.info("Awaiting pipeline execution. Use the left sidebar to configure your target and launch the scraper.")

elif app_mode == "📝 Raw Data Feed":
    st.header("Raw Stream Content Ledger")
    st.caption("This ledger represents the real-time, unstructured data scraped from the Hacker News API, appended with the immediate outputs of our local HuggingFace NLP pipelines. It serves as the manual audit layer before vectorization.")
    
    if st.session_state.active_data is not None:
        raw_df = st.session_state.active_data
        
        st.markdown("### 📊 Current Scrape Telemetry")
        m1, m2, m3 = st.columns(3)
        m1.metric("Target Keyword", st.session_state.current_query)
        m2.metric("Nodes Extracted", len(raw_df))
        
        dominant_vibe = raw_df['Comment Vibe'].mode()[0] if not raw_df.empty else "N/A"
        m3.metric("Dominant Text Vibe", dominant_vibe)
        
        st.markdown("---")
        
        st.dataframe(
            raw_df.drop(columns=['Raw_Confidence']), 
            use_container_width=True, 
            height=500,
            hide_index=True 
        )
    else:
        st.info("No active data in memory. Execute a pipeline run from the sidebar.")

elif app_mode == "🗄️ Historical Warehouse":
    st.header("Relational Infrastructure Browser")
    st.caption("This tab accesses the persistent SQLite database (`insights.db`). This tabular data acts as the ground-truth relational mapping for the mathematical FAISS vector index used by the RAG engine.")
    
    col_ctrl1, col_ctrl2 = st.columns([1, 5])
    with col_ctrl1:
        if st.button("🗑️ Clear Data"):
            conn = sqlite3.connect(DB_FILE)
            conn.execute("DELETE FROM ai_insights")
            conn.commit()
            conn.close()
            st.session_state.active_data = None 
            if os.path.exists(FAISS_INDEX_FILE): os.remove(FAISS_INDEX_FILE)
            if os.path.exists(MAP_FILE): os.remove(MAP_FILE)
            st.success("Database cleared successfully!")
            st.rerun()
            
    st.markdown("---")
    
    try:
        conn = sqlite3.connect(DB_FILE)
        history_df = pd.read_sql_query("SELECT * FROM ai_insights ORDER BY id DESC", conn)
        conn.close()
        
        if not history_df.empty:
            
            st.markdown("### 📈 Warehouse Telemetry")
            hm1, hm2, hm3 = st.columns(3)
            hm1.metric("Total Archived Records", len(history_df))
            hm2.metric("Unique Topics Tracked", history_df['search_query'].nunique())
            
            latest_time = history_df['timestamp'].max()
            time_only = latest_time.split(" ")[1] if isinstance(latest_time, str) else "N/A"
            hm3.metric("Last Database Write", time_only)
            
            st.markdown("<br>", unsafe_allow_html=True)
            topic_counts = history_df['search_query'].value_counts().reset_index()
            topic_counts.columns = ['Tracked Topic', 'Volume of Records']
            fig_history = px.bar(
                topic_counts, x='Tracked Topic', y='Volume of Records', 
                template='plotly_dark',
                color_discrete_sequence=['#C792EA'] 
            )
            fig_history.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=250)
            st.plotly_chart(fig_history, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 🗃️ Raw SQLite Ledger")
            
            csv_payload = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Warehouse to CSV",
                data=csv_payload,
                file_name=f"insightai_warehouse_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            st.dataframe(history_df, use_container_width=True, height=400, hide_index=True)
        else:
            st.info("Database is initialized but currently empty.")
    except Exception as e:
        st.error(f"Failed to access local database: {e}")

elif app_mode == "💬 AI Data Analyst":
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("💬 Intelligence Console (RAG)")
    with col2:
        if st.button("🗑️ Clear History"):
            conn = sqlite3.connect(DB_FILE)
            conn.execute("DELETE FROM chat_sessions WHERE username = ?", (st.session_state.logged_in_user,))
            conn.commit()
            conn.close()
            st.rerun()
            
    st.caption("This interface bypasses standard LLM hallucinations. It embeds your query into a mathematical vector, searches the local FAISS index for the nearest semantic matches, and forces the selected AI model to synthesize an answer strictly from your scraped historical data.")
    st.markdown("---")
            
    if st.session_state.llm_api_key:
        if st.session_state.llm_provider == "Google Gemini":
            genai.configure(api_key=st.session_state.llm_api_key)
            model_engine = genai.GenerativeModel(st.session_state.llm_model)
        else:
            openai_client = OpenAI(api_key=st.session_state.llm_api_key)
        
        conn = sqlite3.connect(DB_FILE)
        history_df = pd.read_sql_query("SELECT role, content FROM chat_sessions WHERE username = ? ORDER BY id ASC", conn, params=(st.session_state.logged_in_user,))
        conn.close()
        
        # --- NEW: Enterprise "Empty State" Terminal UI ---
        if history_df.empty:
            st.markdown(f"""
            <div style='background-color: #1A1A1A; padding: 30px; border-radius: 8px; border: 1px solid #333333; text-align: center; margin-bottom: 20px;'>
                <h2 style='color: #F8DF72; margin-top: 0; font-family: monospace;'>System Online: Ready for Query</h2>
                <p style='color: #888888; font-family: monospace; font-size: 1.1rem;'>
                    [ Database ]: insights.index (FAISS)<br>
                    [ AI Engine ]: {st.session_state.llm_provider} ({st.session_state.llm_model})
                </p>
                <hr style='border-color: #333; margin: 20px 0;'>
                <p style='color: #FFFFFF; font-size: 1.1rem;'>The RAG engine is primed. Try asking a question about your ingested data:</p>
                <ul style='list-style-type: none; padding: 0; color: #82AAFF; font-family: monospace; font-size: 1.1rem;'>
                    <li>> "What are the main complaints regarding the latest release?"</li>
                    <li>> "Summarize the positive feedback from the community."</li>
                    <li>> "Are there any mentions of security vulnerabilities?"</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        for _, msg in history_df.iterrows():
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        user_question = st.chat_input("Ask a question about your collected insights...")
        
        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)
                
            conn = sqlite3.connect(DB_FILE)
            conn.execute("INSERT INTO chat_sessions (username, role, content, timestamp) VALUES (?, ?, ?, ?)",
                         (st.session_state.logged_in_user, "user", user_question, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
            
            with st.chat_message("assistant"):
                if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(MAP_FILE):
                    with st.spinner("Compiling structural database index on disk..."):
                        build_and_save_vector_index()
                        
                if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(MAP_FILE):
                    try:
                        index = faiss.read_index(FAISS_INDEX_FILE)
                        with open(MAP_FILE, "r", encoding="utf-8") as f:
                            documents = [line.strip() for line in f.readlines()]
                            
                        query_vector = vector_embedder.encode([user_question], convert_to_numpy=True)
                        k = min(5, len(documents))
                        distances, indices = index.search(query_vector, k)
                        relevant_context = "\n".join([documents[i] for i in indices[0] if i < len(documents)])
                        
                        prompt = f"""
                        You are an expert Data Analyst working for an enterprise intelligence firm. 
                        Use ONLY the following semantically retrieved database records to answer the user's question.
                        
                        CRITICAL RULE FOR INTEGRITY: When mentioning specific findings, opinions, comments, or headlines from the context data, 
                        you MUST append its structural Source ID in brackets at the end of the sentence, for example: [Source Record ID: X].
                        
                        Relevant Database Records Context:
                        {relevant_context}
                        
                        User Question: "{user_question}"
                        """
                        
                        def stream_llm_response():
                            if st.session_state.llm_provider == "Google Gemini":
                                response_stream = model_engine.generate_content(prompt, stream=True)
                                for chunk in response_stream:
                                    yield chunk.text
                            elif st.session_state.llm_provider == "OpenAI":
                                response_stream = openai_client.chat.completions.create(
                                    model=st.session_state.llm_model,
                                    messages=[{"role": "user", "content": prompt}],
                                    stream=True
                                )
                                for chunk in response_stream:
                                    if chunk.choices[0].delta.content:
                                        yield chunk.choices[0].delta.content
                                
                        output_text = st.write_stream(stream_llm_response())
                        
                        with st.expander("🔍 Review Explanatory Vector Retrieval Sources"):
                            st.write("The pre-computed FAISS index instantly pulled these vector references from disk:")
                            st.text(relevant_context)
                        
                        conn = sqlite3.connect(DB_FILE)
                        conn.execute("INSERT INTO chat_sessions (username, role, content, timestamp) VALUES (?, ?, ?, ?)",
                                     (st.session_state.logged_in_user, "assistant", output_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        conn.commit()
                        conn.close()
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Execution Error: {e}")
                else:
                    st.warning("Data repository is currently dry. Populate it first via the pipeline.")
    else:
        st.warning(f"⚠️ Please provide a valid API Key in the sidebar to activate the {st.session_state.llm_provider} framework.")
