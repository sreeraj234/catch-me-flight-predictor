# 🏃✈️ Catch Me If You Can: Flight Connection Probability Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PySpark](https://img.shields.io/badge/PySpark-Data_Processing-orange.svg)
![Databricks](https://img.shields.io/badge/Databricks-MLflow-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B.svg)

## 📌 Project Overview
**"Catch me if you can..."** – the age-old race through the airport terminal. 

**Catch Me If You Can** is an end-to-end Machine Learning pipeline designed to estimate the likelihood of a passenger successfully making a connecting flight. Originally conceptualized as a winning project at the Databricks x UW Data Science Hackathon, this repository represents the fully engineered, production-ready version.

Using millions of rows of historical aviation data, the system relies on **PySpark** for distributed data processing, **Databricks/MLflow** for model tracking, and a decoupled **FastAPI + Streamlit** architecture for real-time inference.

## 🏗️ Architecture Workflow
1. **Data Ingestion:** Historical flight data from the Bureau of Transportation Statistics (BTS).
2. **Data Engineering (PySpark):** Complex self-joins on distributed clusters to generate valid "passenger itineraries" (Flight A -> Hub -> Flight B).
3. **Machine Learning (Spark MLlib):** Training ensemble models to predict connection success based on layover margins and historical delays.
4. **Model Registry:** Databricks MLflow for tracking parameters and metrics.
5. **Serving Layer:** FastAPI REST endpoint hosting the saved model.
6. **User Interface:** Streamlit web application for interactive predictions.

## 📊 Dataset & Features
**Primary Data Source:** [BTS Airline On-Time Performance Data](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr)

**Key Engineered Features:**
*   `scheduled_layover_time`: Minutes between Flight A's scheduled arrival and Flight B's scheduled departure.
*   `airline_historical_reliability`: 30-day rolling average delay for the specific airline.
*   `hub_congestion_index`: Average delay metrics at the connecting airport for the given hour of the day.
*   `temporal_factors`: Month, Day of Week, Hour of Day (capturing seasonal and daily traffic trends).

**Target Variable:**
*   `1 (Success)`: Flight A Actual Arrival + Minimum Connection Time (MCT) < Flight B Actual Departure.
*   `0 (Missed)`: Passenger failed to make the connection.

## 🧠 Models Evaluated
The project frames this as a binary classification problem (handling class imbalance due to a high success rate in reality).
*   **Baseline:** Logistic Regression (`pyspark.ml.classification.LogisticRegression`)
*   **Ensemble 1:** Random Forest (`RandomForestClassifier`) - *Handles non-linear relationships and interactions well.*
*   **Ensemble 2:** Gradient Boosted Trees (`GBTClassifier`) - *Optimized for maximum ROC-AUC and F1-Score.*

## 🚀 To-Do / Roadmap (Implementation Steps)

### Phase 1: Data Engineering
- [x] Setup Databricks Community Edition workspace.
- [x] Ingest BTS historical flight data (min. 3-6 months) to DBFS.
- [x] Write PySpark self-join logic to simulate connecting flights.
- [x] Engineer target variable (`made_connection`).

### Phase 2: Feature Engineering & Modeling
- [ ] Extract time-based features (layover duration, hour of day).
- [ ] Extract historical reliability features (airline & hub delays).
- [ ] Train Baseline Logistic Regression model.
- [ ] Train GBT / Random Forest model.
- [ ] Log parameters, F1-scores, and ROC-AUC metrics using MLflow.

### Phase 3: API Backend (FastAPI)
- [ ] Export the best-performing model pipeline.
- [ ] Setup a local FastAPI project.
- [ ] Create a `POST /predict` endpoint that accepts flight details and returns a probability score.
- [ ] Test API endpoints using Postman or Python `requests`.

### Phase 4: Frontend & Deployment (Streamlit)
- [ ] Build a Streamlit UI allowing users to input Origin, Hub, Destination, Airlines, and Times.
- [ ] Connect Streamlit UI to the FastAPI backend.
- [ ] Deploy FastAPI backend to Render / Railway.
- [ ] Deploy Streamlit frontend to Streamlit Community Cloud.

## 💻 Local Setup & Installation
*(Instructions to be updated once API and UI are finalized)*

```bash
# Clone the repository
git clone https://github.com/YourUsername/Flight-Connection-Engine.git
cd Flight-Connection-Engine

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install requirements
pip install -r requirements.txt

# Run the FastAPI backend
uvicorn api.main:app --reload

# Run the Streamlit frontend (in a new terminal)
streamlit run app/frontend.py
