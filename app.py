import streamlit as st
import pandas as pd
import mlflow.pyfunc
from databricks import sql
from databricks.sdk import WorkspaceClient
import plotly.graph_objects as go
import os

# --- 1. SETUP & AUTHENTICATION (Official SDK Pattern) ---
st.set_page_config(page_title="Catch Me If You Can", page_icon="✈️", layout="wide")

@st.cache_resource
def load_resources():
    # Use the WorkspaceClient to find the SQL Warehouse automatically
    # This avoids the "NoneType" and "startswith" errors entirely
    w = WorkspaceClient()
    
    # 1. Discover the SQL Warehouse HTTP Path
    # We look for the variable Databricks injected when you added the Resource
    http_path = None
    for key, value in os.environ.items():
        if "HTTP_PATH" in key:
            http_path = value
            break
            
    if not http_path:
        # Fallback: Find the first available Serverless Warehouse if the Resource link failed
        warehouses = list(w.warehouses.list())
        if warehouses:
            http_path = warehouses[0].jdbc_url.split("HttpPath=")[1].split(";")[0]

    # 2. Get Connection Credentials
    host = os.environ.get("DATABRICKS_HOST").replace("https://", "")
    token = os.environ.get("DATABRICKS_TOKEN")

    # 3. Connect to SQL Warehouse for Lookup Data
    connection = sql.connect(
        server_hostname=host,
        http_path=http_path,
        access_token=token
    )
    
    with connection.cursor() as cursor:
        cursor.execute("SELECT Code, Description FROM workspace.flights.AIRPORT_CODES")
        apt_df = pd.DataFrame(cursor.fetchall(), columns=["Code", "Description"])
        
        cursor.execute("SELECT Code, Description FROM workspace.flights.UNIQUE_CARRIERS")
        carrier_df = pd.DataFrame(cursor.fetchall(), columns=["Code", "Description"])

    # 4. Load the GBT Model (Ensure Tracking URI is set for Databricks)
    mlflow.set_tracking_uri("databricks")
    # Replace with your actual GBT Run ID
    run_id = "77a6d076805e4ffd9b8d245b1e069b2e"
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/flight_gbt_pipeline")
    
    return model, apt_df, carrier_df

# --- EXECUTION ---
with st.spinner("Initializing AI Engine and connecting to Unity Catalog..."):
    try:
        model, apt_df, carrier_df = load_resources()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.info("Check if the SQL Warehouse and MLflow Experiment are added in the 'Resources' tab.")
        st.stop()

# Prepare UI Data
apt_df['label'] = apt_df['Description'] + " (" + apt_df['Code'] + ")"
carrier_df['label'] = carrier_df['Description'] + " (" + carrier_df['Code'] + ")"
airport_map = dict(zip(apt_df['label'], apt_df['Code']))
carrier_map = dict(zip(carrier_df['label'], carrier_df['Code']))

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("Search Parameters")
    origin_label = st.selectbox("Origin", sorted(apt_df['label'].tolist()))
    hub_label = st.selectbox("Connection Hub", sorted(apt_df['label'].tolist()), index=1)
    carrier_label = st.selectbox("Airline", sorted(carrier_df['label'].tolist()))
    st.divider()
    travel_month = st.select_slider("Month of Travel", options=list(range(1, 13)), value=6)
    arrival_hour = st.slider("Hub Arrival Hour (24h)", 0, 23, 14)

# --- MAIN CONTENT ---
st.title("🏃✈️ Catch Me If You Can")
st.markdown("#### GBT-Powered Flight Connection Probability")

if st.button("🚀 Run Probability Analysis", use_container_width=True):
    with st.spinner("Simulating 180 layover scenarios..."):
        
        layover_range = list(range(10, 185, 5))
        
        # Create Pandas Input - Column names MUST match your Spark Pipeline training columns
        input_data = pd.DataFrame({
            "OP_UNIQUE_CARRIER_A": [carrier_map[carrier_label]] * len(layover_range),
            "ORIGIN_A": [airport_map[origin_label]] * len(layover_range),
            "DEST_A": [airport_map[hub_label]] * len(layover_range),
            "is_same_airline": [1] * len(layover_range),
            "hub_congestion_hour": [arrival_hour] * len(layover_range),
            "travel_month": [travel_month] * len(layover_range),
            "scheduled_layover_mins": [float(m) for m in layover_range]
        })
        
        # Run Inference
        raw_preds = model.predict(input_data)
        
        # Parse output (Spark Pipelines via PyFunc often return a DataFrame with a 'probability' column)
        if isinstance(raw_preds, pd.DataFrame) and 'probability' in raw_preds.columns:
            probs = [p[1] for p in raw_preds['probability']]
        else:
            probs = raw_preds # Fallback if it returns a flat array

        # --- CDF PLOT ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=layover_range, y=probs,
            mode='lines+markers', line_shape='hv',
            line=dict(color='#00d1ff', width=3),
            fill='tozeroy', fillcolor='rgba(0, 209, 255, 0.1)'
        ))
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Layover Duration (Minutes)",
            yaxis_title="Probability of Success",
            yaxis_range=[0, 1.05]
        )
        st.plotly_chart(fig, use_container_width=True)