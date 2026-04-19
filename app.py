import os
from databricks import sql
from databricks.sdk.core import Config
from databricks.sdk import WorkspaceClient
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- 1. CONFIGURATION CHECKS ---
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."
ENDPOINT_NAME = "flight_prediction_model"

# --- 2. THEME & UI CONFIG ---
st.set_page_config(page_title="Catch Me If You Can | Flight Engine", page_icon="✈️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #ffffff; }
    .main-header { font-size: 48px; font-weight: 800; color: #00d1ff; text-align: center; margin-bottom: 0px; }
    .sub-header { font-size: 16px; text-align: center; color: #8b949e; margin-bottom: 30px; }
    .stSelectbox label, .stSlider label { color: #00d1ff !important; font-weight: bold; }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATABRICKS RESOURCE LOADERS ---
@st.cache_data(ttl=3600)
def sqlQuery(query: str) -> pd.DataFrame:
    cfg = Config()
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
        credentials_provider=lambda: cfg.authenticate
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()

@st.cache_resource
def get_workspace_client():
    return WorkspaceClient()

@st.cache_resource
def ensure_endpoint_started():
    """Start the endpoint if it's stopped, wait until ready"""
    w = get_workspace_client()
    endpoint_name = ENDPOINT_NAME

    try:
        endpoint = w.serving_endpoints.get(endpoint_name)

        # Check if endpoint is stopped
        if endpoint.state.ready == "NOT_READY":
            st.info(f"🔄 Starting endpoint {endpoint_name}... This may take 2-3 minutes.")
            
            # Start by updating the endpoint (triggers start)
            w.serving_endpoints.update_config(
                name=endpoint_name,
                served_entities=endpoint.config.served_entities
            )
            
            # Wait for endpoint to become ready
            import time
            max_wait = 300  # 5 minutes timeout
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                endpoint = w.serving_endpoints.get(endpoint_name)
                if endpoint.state.ready == "READY":
                    st.success(f"✅ Endpoint {endpoint_name} is ready!")
                    return True
                time.sleep(10)  # Check every 10 seconds
            
            st.warning("Endpoint is taking longer than expected to start.")
            return False
        else:
            return True
            
    except Exception as e:
        st.error(f"Error checking endpoint: {str(e)}")
        return False

# --- 4. FETCH DATA ---
try:
    # Start endpoint if stopped
    ensure_endpoint_started()

    with st.spinner("Connecting to Databricks SQL Warehouse..."):
        apt_df = sqlQuery("""
            SELECT DISTINCT a.Code, a.Name 
            FROM workspace.flights.AIRPORT_CODES a
            JOIN (SELECT DISTINCT ORIGIN_A FROM workspace.flights.ml_train_gold) t ON a.Code = t.ORIGIN_A
        """)
        
        carrier_df = sqlQuery("""
            SELECT DISTINCT c.Code, c.Description 
            FROM workspace.flights.UNIQUE_CARRIERS c
            JOIN (SELECT DISTINCT OP_UNIQUE_CARRIER_A FROM workspace.flights.ml_train_gold) t ON c.Code = t.OP_UNIQUE_CARRIER_A
        """)
        
except Exception as e:
    st.error("Failed to load Databricks resources. Did you grant Unity Catalog permissions to the App Service Principal?")
    st.exception(e)
    st.stop()

# Build mapping dictionaries for the UI
apt_df['label'] = apt_df['Code'] + " - " + apt_df['Name']
airport_map = pd.Series(apt_df['Code'].values, index=apt_df['label']).to_dict()
airport_labels = sorted(apt_df['label'].tolist())

carrier_df['label'] = carrier_df['Code'] + " - " + carrier_df['Description']
carrier_map = pd.Series(carrier_df['Code'].values, index=carrier_df['label']).to_dict()
carrier_labels = sorted(carrier_df['label'].tolist())

# --- 5. FRONTEND UI ---
st.markdown('<p class="main-header">🏃✈️ Catch Me If You Can</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI Engine for Connection Risk Assessment</p>', unsafe_allow_html=True)

with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        origin_label = st.selectbox("Departure Airport", airport_labels)
    with c2:
        default_hub = min(5, len(airport_labels)-1) if airport_labels else 0
        hub_label = st.selectbox("Connection Hub (Via)", airport_labels, index=default_hub)
    with c3:
        airline_label = st.selectbox("Airline Carrier", carrier_labels)

    c4, c5, c6 = st.columns([1, 1, 2])
    with c4:
        month = st.selectbox("Month of Travel", list(range(1, 13)), index=5)
    with c5:
        hour = st.selectbox("Arrival Hour (24h)", list(range(0, 24)), index=14)
    with c6:
        st.write("")
        analyze_btn = st.button("🚀 Run AI Probability Analysis", use_container_width=True)

st.divider()

# --- 6. INFERENCE EXECUTION ---
if analyze_btn:
    with st.spinner("Processing simulation points..."):
        layover_range = list(range(5, 185, 2))
        
        # Translate UI labels back to pure Codes for the ML Model
        input_data = pd.DataFrame({
            "OP_UNIQUE_CARRIER_A":[carrier_map[airline_label]] * len(layover_range),
            "ORIGIN_A":[airport_map[origin_label]] * len(layover_range),
            "DEST_A":[airport_map[hub_label]] * len(layover_range),
            "is_same_airline": [1] * len(layover_range),
            "hub_congestion_hour": [hour] * len(layover_range),
            "travel_month": [month] * len(layover_range),
            "scheduled_layover_mins":[float(m) for m in layover_range]
        })
        
        # Run Databricks Model Serving Inference using SDK
        try:
            w = get_workspace_client()
            response = w.serving_endpoints.query(
                name=ENDPOINT_NAME,
                dataframe_records=input_data.to_dict(orient='records')
            )
            
            # Extract predictions from SDK response
            predictions = response.predictions
            
            if isinstance(predictions, list) and len(predictions) > 0:
                if isinstance(predictions[0], dict) and 'probability' in predictions[0]:
                    prob_arrays = [p['probability'] for p in predictions]
                    if isinstance(prob_arrays[0], list) and len(prob_arrays[0]) > 1:
                        probs = [p[1] for p in prob_arrays]
                    else:
                        probs = prob_arrays
                elif isinstance(predictions[0], (list, tuple)) and len(predictions[0]) > 1:
                    probs = [p[1] for p in predictions]
                elif isinstance(predictions[0], (int, float)):
                    probs = predictions
                else:
                    st.error(f"Unexpected prediction format: {type(predictions[0])}")
                    st.write("Sample prediction:", predictions[0])
                    st.stop()
            else:
                st.error("Predictions list is empty or invalid")
                st.stop()
                
        except Exception as e:
            st.error(f"Model prediction error: {str(e)}")
            st.exception(e)
            st.stop()

        # Render Plotly Chart
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
        
        # AI Insight Generator
        try:
            safe_time = next((x for x, y in zip(layover_range, probs) if y > 0.80), 90)
            st.info(f"**💡 AI Insight:** At {airport_map[hub_label]}, an 80% success rate is reached at **{safe_time} minutes**.")
        except:
            st.warning("⚠️ **AI Insight:** High-risk connection profile detected.")
