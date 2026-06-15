# 🧠 InsightAI: Enterprise-Grade RAG & Sentiment Analytics

InsightAI is a sophisticated Retrieval-Augmented Generation (RAG) platform designed to synthesize unstructured web data into actionable business intelligence. By integrating real-time API ingestion with local NLP pipelines and LLM-driven synthesis, InsightAI solves the critical "hallucination problem" inherent in standard generative models.

---

## 🚀 Executive Summary
In an era of information overload, data-driven decision-making requires speed and accuracy. InsightAI bridges the gap between raw, noisy forum data and clean, evidence-based insights. Our platform empowers analysts to interrogate live tech discussions with high-confidence, source-cited responses while maintaining complete data transparency.

---

## ⚙️ Core Technical Architecture

### 1. Dynamic Data Ingestion Engine
* **Concurrency:** Utilizes `ThreadPoolExecutor` for high-throughput, multi-threaded API communication, ensuring sub-second data retrieval.
* **Pipeline Integrity:** Implements a robust ETL process that cleans, structures, and persists raw data into a relational SQLite ledger before vectorization.

### 2. Multi-Stage NLP & Vectorization
* **Semantic Intelligence:** Employs dual-stream Transformer pipelines (RoBERTa & DistilBERT) to extract concurrent sentiment and emotional sentiment.
* **Vector Database:** Leverages `Meta FAISS` and `sentence-transformers` to enable high-speed semantic similarity search, transforming unstructured text into mathematical vector spaces.

### 3. Hallucination-Free Query Engine (RAG)
* **Contextual Synthesis:** Instead of relying on pre-trained parametric memory, our AI Data Analyst employs a strict RAG retrieval pattern.
* **Source Attribution:** The system forces the LLM to perform structural citation using unique record IDs, ensuring every data-driven claim is backed by ground-truth evidence.

---

## 📊 Feature Highlights
* **Predictive Sentiment Analytics:** Provides historical sentiment trend mapping and algorithmic trajectory forecasting.
* **Enterprise Security:** Implements secure user authentication and configurable API gateway integration for Gemini and OpenAI models.
* **Audit-Ready Dashboards:** Features high-contrast, "Carbon-style" visualizations that prioritize clarity for stakeholder reporting.

---

## 🛠️ Tech Stack
* **Orchestration:** Streamlit (Python-native UI)
* **Inference:** HuggingFace `transformers` (Local), OpenAI/Gemini (Generative)
* **Persistence:** SQLite3 & FAISS (Local Vector Ledger)
* **Visualization:** Plotly & Pandas

---

## 🚦 Deployment & Usage

### Setup
1. Clone the repository: `git clone https://github.com/prahlad200/Insight-AI.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the environment: `streamlit run app.py`

### Configuration
For consistent UI styling, ensure the `.streamlit/config.toml` file is present in your root directory to initialize our high-contrast enterprise theme.

