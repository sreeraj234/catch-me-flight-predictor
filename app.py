import streamlit as st
import pandas as pd
import mlflow.pyfunc
from databricks import sql
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Catch Me If You Can", page_icon="✈️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .main-header { font-size: 42px; font-weight: 800; color: #00d1ff; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_resources():
    # CLEAN THE HOSTNAME: The connector fails if it sees 'https://'
    host = os.getenv("DATABRICKS_HOST", "").replace("https://", "")
    token = os.getenv("DATABRICKS_TOKEN")
    http_path = os.getenv("DATABRICKS_SQL_HTTP_PATH")

    # 1. Connect to SQL for Lookups
    connection = sql.connect(
        server_hostname=host,
        http_path=http_path,
        access_token=token
    )
    
    with connection.cursor() as cursor:
        cursor.execute("SELECT Code, Description FROM workspace.flights.AIRPORT_CODES")
        airports = pd.DataFrame(cursor.fetchall(), columns=["Code", "Description"])
        
        cursor.execute("SELECT Code, Description FROM workspace.flights.UNIQUE_CARRIERS")
        carriers = pd.DataFrame(cursor.fetchall(), columns=["Code", "Description"])
        
    # 2. Load Model via PyFunc
    # Replace this with your actual GBT Run ID
    run_id = "77a6d076805e4ffd9b8d245b1e069b2e"
    # Note: MLflow set_tracking_uri is handled automatically in Databricks Apps
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/flight_gbt_pipeline")
    
    return model, airports, carriers

# --- EXECUTION ---
try:
    model, apt_df, carrier_df = get_resources()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# Prepare Dropdowns
apt_df['label'] = apt_df['Description'] + " (" + apt_df['Code'] + ")"
carrier_df['label'] = carrier_df['Description'] + " (" + carrier_df['Code'] + ")"
airport_map = dict(zip(apt_df['label'], apt_df['Code']))
carrier_map = dict(zip(carrier_df['label'], carrier_df['Code']))

# --- SIDEBAR ---
with st.sidebar:
    st.header("Flight Search")
    origin_label = st.selectbox("Origin", sorted(apt_df['label'].tolist()))
    hub_label = st.selectbox("Hub (Via)", sorted(apt_df['label'].tolist()), index=1)
    carrier_label = st.selectbox("Airline", sorted(carrier_df['label'].tolist()))
    st.divider()
    travel_month = st.select_slider("Month", options=list(range(1, 13)), value=6)
    arrival_hour = st.slider("Arrival Hour", 0, 23, 14)

# --- MAIN UI ---
st.markdown('<p class="main-header">Catch Me If You Can</p>', unsafe_allow_html=True)

if st.button("🚀 Analyze Connection Success", use_container_width=True):
    with st.spinner("Calculating probability curve..."):
        
        layover_range = list(range(10, 185, 5))
        
        # Construct Pandas Input for the Model
        input_data = pd.DataFrame({
            "OP_UNIQUE_CARRIER_A": [carrier_map[carrier_label]] * len(layover_range),
            "ORIGIN_A": [airport_map[origin_label]] * len(layover_range),
            "DEST_A": [airport_map[hub_label]] * len(layover_range),
            "is_same_airline": [1] * len(layover_range),
            "hub_congestion_hour": [arrival_hour] * len(layover_range),
            "travel_month": [travel_month] * len(layover_range),
            "scheduled_layover_mins": [float(m) for m in layover_range]
        })
        
        # PyFunc predict on a Spark Pipeline usually returns a DataFrame 
        # that includes the 'probability' column as a list/array
        predictions_output = model.predict(input_data)
        
        # CDF extraction logic
        # If the output is a DataFrame with a 'probability' column
        if isinstance(predictions_output, pd.DataFrame) and 'probability' in predictions_output.columns:
            # Spark probabilities are [prob_fail, prob_success]
            probs = [p[1] for p in predictions_output['probability']]
        else:
            # Fallback if the model only returns classes
            probs = predictions_output 

        # --- PLOTLY ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=layover_range, y=probs,
            mode='lines+markers', line_shape='hv',
            line=dict(color='#00d1ff', width=3),
            fill='tozeroy', fillcolor='rgba(0, 209, 255, 0.1)'
        ))
        fig.update_layout(
            title=f"Probability Curve: {airport_map[origin_label]} ➔ {airport_map[hub_label]}",
            template="plotly_dark",
            xaxis_title="Layover Duration (Minutes)",
            yaxis_title="Success Probability",
            yaxis_range=[0, 1.05]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("Analysis complete. The curve represents your likelihood of making the connection as layover time increases.")