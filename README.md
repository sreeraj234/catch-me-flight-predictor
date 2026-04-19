# 🏃✈️ Catch Me If You Can: Flight Connection Probability Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PySpark](https://img.shields.io/badge/PySpark-Data_Processing-orange.svg)
![Databricks](https://img.shields.io/badge/Databricks-Unity_Catalog-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-blue.svg)

## 📌 Project Overview
**"Catch me if you can..."** – the age-old race through the airport terminal. 

**Catch Me If You Can** is an end-to-end Machine Learning pipeline designed to estimate the likelihood of a passenger successfully making a connecting flight. Originally conceptualized as a winning project at the **Databricks x UW Data Science Hackathon**, this repository represents a scaled, production-ready version capable of processing nearly **half a billion simulated itineraries**.

By leveraging **PySpark** on **Databricks Unity Catalog**, the engine analyzes historical flight patterns, airport congestion, and seasonal weather trends to provide real-time connection probabilities via a **Probability Distribution Function** visualization.

## 🚀 Key Highlights
*   **Scale:** Engineered a distributed pipeline that processed **7 Million raw flights** and executed an optimized non-equi self-join to generate **499 Million connecting flight pairs**.
*   **Methodology:** Implemented a strict **Out-of-Time (OOT) Validation** strategy, training on the full year of 2024 and testing on a pristine Q1 2025 holdout set.
*   **Performance:** The champion **Gradient Boosted Trees (GBT)** model achieved an **AUC-ROC of 0.8234** and **98% Precision** on unseen 2025 data.

## 🏗️ Architecture & Pipeline
1.  **Data Engineering (PySpark & Unity Catalog):**
    *   Ingested 12+ months of BTS Airline On-Time Performance data.
    *   Resolved "midnight rollover" timestamp issues using continuous mathematical reconstruction.
    *   Optimized a high-volume self-join using day-partitioned hash joins to prevent Cartesian product overhead.
2.  **Feature Engineering:**
    *   **Categorical:** One-Hot Encoding for `Airline`, `Origin`, and `Hub`.
    *   **Temporal:** `Hub_Congestion_Hour` and `Travel_Month` (Seasonality).
    *   **Interaction:** `Is_Same_Airline` (Proxy for terminal transfer friction).
3.  **Model Governance (MLflow):**
    *   Tracked experiments and hyperparameter tuning using MLflow.
    *   Managed model artifacts and serialization via Unity Catalog Volumes.
4.  **Serving Layer (In-Progress):**
    *   Decoupled FastAPI backend hosting the Spark ML Pipeline.
    *   Streamlit frontend for user-facing probability forecasting and CDF plotting.

## 📊 Model Performance (2025 Holdout Set)
| Metric | Random Forest | **GBT (Champion)** |
| :--- | :--- | :--- |
| **AUC-ROC** | 0.7880 | **0.8234** |
| **F1-Score** | 0.7925 | **0.8372** |
| **Precision** | 0.9504 | **0.9868** |
| **Recall** | 0.7030 | **0.7270** |

## 🚀 Implementation Roadmap

### Phase 1: Data Engineering & Scaling
- [x] Setup Databricks Unity Catalog & Volume architecture.
- [x] Build mathematically robust timestamp reconstruction logic.
- [x] Execute optimized self-join (499M connections generated).
- [x] Materialize "Gold" Feature Tables for ML training.

### Phase 2: Distributed Modeling
- [x] Implement Stratified Undersampling to handle 96/4 class imbalance.
- [x] Build Spark ML Pipeline (StringIndexer -> OHE -> Assembler).
- [x] Train and evaluate Random Forest vs. Gradient Boosted Trees.
- [x] Log model artifacts and metrics to MLflow Registry.

### Phase 3: Serving & Deployment (Current)
- [x] Export Spark ML Pipeline for local inference.
- [x] Build **Streamlit** UI with interactive CDF charts.
- [x] Containerize and deploy to cloud (Render/Streamlit Cloud/Databricks App).

## 💻 Local Setup
```bash
git clone https://github.com/YourUsername/Catch-Me-If-You-Can.git
cd Catch-Me-If-You-Can
pip install -r requirements.txt
streamlit run app.py
```

---
*Developed by Sreeraj Parakkat — University of Washington, MS Data Science.*

***