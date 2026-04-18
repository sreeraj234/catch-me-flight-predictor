import os
import streamlit as st
import pandas as pd
import mlflow.pyfunc
from databricks import sql, WorkspaceClient
import plotly.graph_objects as go


# --- 1. THEME & UI CONFIG ---
st.set_page_config(page_title="NextGate AI | Flight Engine", page_icon="✈️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #ffffff; }
    .main-header { font-size: 48px; font-weight: 800; color: #00d1ff; text-align: center; margin-bottom: 0px; }
    .sub-header { font-size: 16px; text-align: center; color: #8b949e; margin-bottom: 30px; }
    .stSelectbox label, .stSlider label { color: #00d1ff !important; font-weight: bold; }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSET LOADER (Production Ready) ---
@st.cache_resource
def load_production_resources():
    # The WorkspaceClient is the "Master Key"
    # It automatically finds DATABRICKS_HOST, CLIENT_ID, and CLIENT_SECRET
    w = WorkspaceClient()
    
    # Get the host (cleaning it for the SQL connector)
    host = w.config.host.replace("https://", "")
    
    # Get the Token dynamically from the Service Principal
    # This replaces the need for a hardcoded DATABRICKS_TOKEN
    token = w.config.authenticate() 

    # 1. Discover the SQL Warehouse Path from Resources
    http_path = os.environ.get("DATABRICKS_SQL_HTTP_PATH")
    if not http_path:
        # Fallback: Loop through env vars to find the one injected by the Resource link
        for key, value in os.environ.items():
            if "HTTP_PATH" in key:
                http_path = value
                break

    # 2. Connect to SQL via the SDK-provided token
    connection = sql.connect(
        server_hostname=host,
        http_path=http_path,
        access_token=token
    )
    
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT DISTINCT a.Code, a.Name 
            FROM workspace.flights.AIRPORT_CODES a
            JOIN (SELECT DISTINCT ORIGIN_A FROM workspace.flights.ml_train_gold) t ON a.Code = t.ORIGIN_A
        """)
        apt_df = pd.DataFrame(cursor.fetchall(), columns=["Code", "Name"])
        
        cursor.execute("""
            SELECT DISTINCT c.Code, c.Description 
            FROM workspace.flights.UNIQUE_CARRIERS c
            JOIN (SELECT DISTINCT OP_UNIQUE_CARRIER_A FROM workspace.flights.ml_train_gold) t ON c.Code = t.OP_UNIQUE_CARRIER_A
        """)
        carrier_df = pd.DataFrame(cursor.fetchall(), columns=["Code", "Description"])
    connection.close()

    # 3. Load MLflow Model
    # MLflow 2.14+ automatically uses the CLIENT_ID/SECRET if tracking_uri is 'databricks'
    mlflow.set_tracking_uri("databricks")
    run_id = os.environ.get("MLFLOW_RUN_ID")
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/flight_gbt_pipeline")
    
    return model, apt_df, carrier_df

# --- EXECUTION ---
try:
    with st.spinner("Authenticating with Databricks Service Principal..."):
        model, apt_df, carrier_df = load_production_resources()
except Exception as e:
    st.error(f"Authentication/Resource Error: {e}")
    st.stop()

# --- 4. FRONTEND UI ---
st.markdown('<p class="main-header">🏃✈️ Catch Me If You Can</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Distributed AI Engine for Connection Risk Assessment</p>', unsafe_allow_html=True)

with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        origin_label = st.selectbox("Departure Airport", sorted(apt_df['label'].tolist()))
    with c2:
        hub_label = st.selectbox("Connection Hub (Via)", sorted(apt_df['label'].tolist()), index=min(5, len(apt_df)-1))
    with c3:
        airline_label = st.selectbox("Airline Carrier", sorted(carrier_df['label'].tolist()))

    c4, c5, c6 = st.columns([1, 1, 2])
    with c4:
        month = st.selectbox("Month of Travel", list(range(1, 13)), index=5)
    with c5:
        hour = st.selectbox("Arrival Hour (24h)", list(range(0, 24)), index=14)
    with c6:
        st.write("") 
        analyze_btn = st.button("🚀 Run AI Probability Analysis", use_container_width=True)

st.divider()

if analyze_btn:
    with st.spinner("Processing simulation points..."):
        # Prepare Batch Data for smooth CDF
        layover_range = list(range(5, 185, 2))
        input_data = pd.DataFrame({
            "OP_UNIQUE_CARRIER_A": [carrier_map[airline_label]] * len(layover_range),
            "ORIGIN_A": [airport_map[origin_label]] * len(layover_range),
            "DEST_A": [airport_map[hub_label]] * len(layover_range),
            "is_same_airline": [1] * len(layover_range),
            "hub_congestion_hour": [hour] * len(layover_range),
            "travel_month": [month] * len(layover_range),
            "scheduled_layover_mins": [float(m) for m in layover_range]
        })
        
        # Inference
        preds = model.predict(input_data)
        
        # Robust Probability Extraction
        if isinstance(preds, pd.DataFrame) and 'probability' in preds.columns:
            probs = [p[1] for p in preds['probability']]
        elif hasattr(preds, 'shape') and len(preds.shape) > 1:
            probs = preds[:, 1]
        else:
            probs = preds

        # CDF Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=layover_range, y=probs,
            mode='lines',
            line=dict(color='#00d1ff', width=4, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(0, 209, 255, 0.15)',
            name='Confidence'
        ))
        
        fig.update_layout(
            title=dict(text=f"Connection Success Probability: {airport_map[origin_label]} ➔ {airport_map[hub_label]}", font=dict(size=20)),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Layover Duration (Minutes)", gridcolor='#23262c', range=[0, 180]),
            yaxis=dict(title="Probability", tickformat=".0%", range=[0, 1.05], gridcolor='#23262c'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        try:
            safe_time = next((x for x, y in zip(layover_range, probs) if y > 0.80), 90)
            st.info(f"**AI Insight:** At {airport_map[hub_label]}, an 80% success rate is reached at **{safe_time} minutes**.")
        except:
            st.warning("AI Insight: High-risk connection profile detected.")

st.markdown("---")
st.caption("Trained on 499M flights via Databricks.")

with st.expander("🛠️ System Diagnostics (Debug Only)"):
    st.write(f"**Host Found:** {bool(os.environ.get('DATABRICKS_HOST'))}")
    st.write(f"**Token Found:** {bool(os.environ.get('DATABRICKS_TOKEN'))}")
    st.write(f"**SQL Path Found:** {bool(os.environ.get('DATABRICKS_SQL_HTTP_PATH'))}")
    st.write(f"**Model ID Found:** {bool(os.environ.get('MLFLOW_RUN_ID'))}")
    
    # This prints the keys of all variables to the logs
    print(f"Available Env Vars: {list(os.environ.keys())}")