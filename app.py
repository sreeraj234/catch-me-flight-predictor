import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Catch Me If You Can",
    page_icon="🏃✈️",
    layout="wide"
)

# --- CUSTOM CSS (To match the clean Hackathon look) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_name_with_display=True)

# --- HEADER ---
st.title("🏃✈️ Catch Me If You Can")
st.markdown("### Search Flights & Connection Setup")
st.write("Choose a flight, then we compute the probability of making your connection.")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Search Parameters")
    
    # Placeholders for now - we will connect these to your BTS lookup tables later
    origin = st.selectbox("Origin Airport", ["ORD (Chicago)", "SEA (Seattle)", "JFK (New York)"])
    hub = st.selectbox("Connection Airport (Hub)", ["DTW (Detroit)", "ATL (Atlanta)", "DEN (Denver)"])
    airline = st.selectbox("Airline", ["Delta (DL)", "United (UA)", "American (AA)"])
    
    st.divider()
    
    st.subheader("Time Context")
    month = st.slider("Month of Travel", 1, 12, 6)
    hour = st.slider("Scheduled Arrival Hour at Hub", 0, 23, 12)
    
    st.info("The model accounts for seasonal weather and airport-specific congestion.")

# --- MAIN UI LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Connection Details")
    st.write(f"**Route:** {origin} ➔ {hub}")
    st.write(f"**Carrier:** {airline}")
    
    # Mock 'Flight List' like your screenshot
    mock_data = {
        "Flight #": ["1594"],
        "Dep Time": ["08:49"],
        "Arr Time": ["11:10"],
        "Status": ["On Time"]
    }
    st.table(pd.DataFrame(mock_data))

# --- CDF CALCULATION (MOCK LOGIC) ---
# This simulates the "looping" through layover minutes
if st.button("📈 Show CDF (Connection Probability)"):
    
    # Simulate x-axis: Layover time from 0 to 180 minutes
    x_layover = np.arange(0, 181, 5)
    
    # Simulate y-axis: A 'Stepped' CDF curve using a sigmoid function
    # In the real version, this will be 180 calls to your GBT model
    y_prob = 1 / (1 + np.exp(-(x_layover - 45) / 10)) 
    
    # --- PLOTLY CHART ---
    fig = go.Figure()

    # The CDF Line
    fig.add_trace(go.Scatter(
        x=x_layover, 
        y=y_prob, 
        mode='lines+markers', 
        name='Probability',
        line_shape='hv', # 'hv' creates the 'Step' look from your screenshot
        line=dict(color='#1f77b4', width=3)
    ))

    # Add a vertical dashed line at a 'standard' 45 min connection
    fig.add_vline(x=45, line_dash="dash", line_color="red", annotation_text="Typical Gate Close")

    fig.update_layout(
        title=f"CDF for Connection at {hub}",
        xaxis_title="Layover Duration (C minutes)",
        yaxis_title="Probability of Success (%)",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Success Message
    st.success(f"Based on historical data, a 60-minute layover has a **{y_prob[12]:.1%}** chance of success.")

else:
    st.info("Select your flight details and click the button to generate the probability curve.")