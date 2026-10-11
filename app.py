import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import joblib

from src.data import fetch_live_city_data, build_historical_dataset
from src.model import train_or_load_model, evaluate_model, predict_traffic

st.set_page_config(page_title="UrbanPulse AI", page_icon="🌆", layout="wide", initial_sidebar_state="expanded")

# ---------- Design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 8% 0%, #172c48 0, #0b1220 34%, #090f1a 100%); color: #e8eef8; }
[data-testid="stSidebar"] { background: #0b1423; border-right: 1px solid #22334a; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f3f7ff; }
.hero { padding: 1.35rem 1.6rem; border-radius: 22px; background: linear-gradient(120deg, rgba(22,48,76,.95), rgba(19,34,54,.85)); border: 1px solid #2c4764; margin-bottom: 1.2rem; }
.eyebrow { color: #61e6c4; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; }
.hero-title { font-family: 'Space Grotesk', sans-serif; font-size: 2.45rem; line-height: 1.05; font-weight: 700; margin: .4rem 0 .5rem; }
.hero-sub { color: #a8bbd2; font-size: 1rem; max-width: 850px; }
.metric-card { padding: 1rem 1.1rem; border-radius: 16px; background: linear-gradient(145deg, #142238, #101a2b); border: 1px solid #263b55; min-height: 118px; }
.metric-label { color: #a4b7cc; font-size: .78rem; text-transform: uppercase; letter-spacing: .09em; }
.metric-value { color: #f2f7ff; font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; margin-top: .25rem; }
.metric-note { color: #61e6c4; font-size: .78rem; margin-top: .2rem; }
.section-label { color: #61e6c4; text-transform: uppercase; font-size: .72rem; font-weight: 700; letter-spacing: .15em; margin: .6rem 0 .25rem; }
.small-muted { color: #9db0c7; font-size: .86rem; }
div[data-testid="stMetric"] { background: #142238; padding: 1rem; border: 1px solid #263b55; border-radius: 14px; }
div[data-testid="stMetricLabel"] { color: #a4b7cc; }
div[data-testid="stMetricValue"] { color: #f2f7ff; }
.stButton>button { border-radius: 10px; border: 1px solid #61e6c4; background: #143c3b; color: #dffff6; font-weight: 600; }
.stButton>button:hover { background: #1b5650; border-color: #8fffe3; color: white; }
hr { border-color: #24364c; }
</style>
""", unsafe_allow_html=True)

CITY_COORDS = {
    "Bengaluru": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Pune": (18.5204, 73.8567),
}

with st.sidebar:
    st.markdown("## 🌆 UrbanPulse AI")
    st.caption("CITY INTELLIGENCE • LIVE SIGNALS")
    city = st.selectbox("City", list(CITY_COORDS.keys()), index=0)
    page = st.radio("Workspace", ["City Overview", "Explore Data", "Prediction Lab", "Model Evaluation"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Data status**")
    st.caption("Live weather + air quality via Open-Meteo")
    st.caption("Traffic & energy are simulated indicators")
    st.caption("Updated when you refresh the dashboard")
    refresh = st.button("↻ Refresh live data", use_container_width=True)
    st.divider()
    st.caption("Built with Python · Streamlit · Plotly · scikit-learn")

lat, lon = CITY_COORDS[city]
live = fetch_live_city_data(city, lat, lon)
historical = build_historical_dataset(city, live)
model, model_source = train_or_load_model(city, historical)
evaluation = evaluate_model(model, historical)
predicted_traffic = predict_traffic(model, live)

now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y · %I:%M %p IST")

st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Urban intelligence platform / {city}</div>
  <div class="hero-title">See the city.<br><span style="color:#61e6c4">Anticipate what's next.</span></div>
  <div class="hero-sub">A compact city digital-twin prototype combining live environmental signals, simulated urban operations, exploratory analytics, and a machine-learning traffic indicator.</div>
  <div style="margin-top:1rem;color:#9db0c7;font-size:.82rem">Last checked: {now} &nbsp; · &nbsp; Model: {model_source}</div>
</div>
""", unsafe_allow_html=True)

if live.get("api_error"):
    st.warning("Live API data could not be loaded. The dashboard is using clearly labelled fallback values; check your internet connection and try refreshing.")

if page == "City Overview":
    st.markdown('<div class="section-label">Live city pulse</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Temperature</div><div class="metric-value">{live["temperature"]:.1f}°C</div><div class="metric-note">Live weather signal</div></div>', unsafe_allow_html=True)
    with c2:
        aqi = live.get("aqi")
        aqi_text = f"{aqi:.0f}" if aqi is not None else "N/A"
        st.markdown(f'<div class="metric-card"><div class="metric-label">European AQI</div><div class="metric-value">{aqi_text}</div><div class="metric-note">Live air-quality signal</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Traffic index</div><div class="metric-value">{predicted_traffic:.0f}/100</div><div class="metric-note">Model estimate · not a traffic feed</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Energy index</div><div class="metric-value">{live["energy_index"]:.0f}/100</div><div class="metric-note">Simulated operational signal</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">City signals at a glance</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35,1])
    with left:
        hours = historical.tail(48).copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours["timestamp"], y=hours["traffic_index"], name="Traffic index", line=dict(color="#61e6c4", width=3), fill="tozeroy", fillcolor="rgba(97,230,196,.09)"))
        fig.add_trace(go.Scatter(x=hours["timestamp"], y=hours["energy_index"], name="Energy index", line=dict(color="#8aa7ff", width=2)))
        fig.update_layout(title="Operational pulse · recent sample window", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=360, margin=dict(l=10,r=10,t=55,b=10), legend=dict(orientation="h", y=1.08), xaxis_title="", yaxis_title="Index (0–100)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=predicted_traffic, number={"suffix":"/100"}, title={"text":"Estimated traffic pressure"}, gauge={"axis":{"range":[0,100]}, "bar":{"color":"#61e6c4"}, "bgcolor":"#142238", "bordercolor":"#2b425d", "steps":[{"range":[0,35],"color":"#183b3b"},{"range":[35,70],"color":"#4b4026"},{"range":[70,100],"color":"#4a2936"}]}))
        gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=360, margin=dict(l=20,r=20,t=55,b=10))
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown('<div class="small-muted">Traffic is a modelled index trained on generated sample data. It is not an official traffic measurement.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Environmental snapshot</div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.metric("Relative humidity", f'{live["humidity"]:.0f}%')
    b.metric("Wind speed", f'{live["wind_speed"]:.1f} km/h')
    c.metric("Rainfall", f'{live["rainfall"]:.1f} mm')

    st.info("Prototype note: weather and European AQI are fetched from Open-Meteo. Traffic and energy indicators are synthetic demonstration data, designed to show the analytics workflow—not to represent official city operations.")

elif page == "Explore Data":
    st.markdown("## Explore urban signals")
    st.markdown('<div class="small-muted">Filter and inspect the generated sample history. Use this view to explore trends and export a CSV for further analysis.</div>', unsafe_allow_html=True)
    d1,d2 = st.columns(2)
    with d1:
        metric = st.selectbox("Signal", ["traffic_index","energy_index","temperature","humidity","aqi"])
    with d2:
        window = st.slider("Hours to display", 24, 168, 72, step=24)
    df = historical.tail(window).copy()
    fig = px.area(df, x="timestamp", y=metric, title=f"{metric.replace('_',' ').title()} · last {window} sample hours")
    fig.update_traces(line_color="#61e6c4", fillcolor="rgba(97,230,196,.13)")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("⬇ Download filtered CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"urbanpulse_{city.lower()}_sample.csv", mime="text/csv")

elif page == "Prediction Lab":
    st.markdown("## Prediction lab")
    st.markdown('<div class="small-muted">Change the scenario inputs to estimate a traffic-pressure index. This is a demonstration model trained on synthetic historical records.</div>', unsafe_allow_html=True)
    p1,p2,p3 = st.columns(3)
    with p1:
        hour = st.slider("Hour of day", 0, 23, datetime.now(ZoneInfo("Asia/Kolkata")).hour)
        weekday = st.selectbox("Day type", ["Weekday","Weekend"])
    with p2:
        temperature = st.slider("Temperature (°C)", 10.0, 45.0, float(live["temperature"]), 0.5)
        rain = st.slider("Rainfall scenario (mm)", 0.0, 40.0, float(live["rainfall"]), 0.5)
    with p3:
        event = st.selectbox("City activity", ["Normal","Concert / event","Festival","Metro disruption"])
        aqi_input = st.slider("European AQI (scenario)", 0, 200, int(live["aqi"] or 50))
    event_factor = {"Normal":0,"Concert / event":18,"Festival":13,"Metro disruption":22}[event]
    scenario = {"hour":hour,"is_weekend":int(weekday=="Weekend"),"temperature":temperature,"rainfall":rain,"aqi":aqi_input,"event_factor":event_factor}
    prediction = predict_traffic(model, scenario)
    st.markdown(f"""
    <div class="hero" style="margin-top:1rem">
      <div class="eyebrow">Scenario result</div>
      <div class="hero-title" style="font-size:2rem">{prediction:.0f} <span style="font-size:1rem;color:#a8bbd2">/ 100 traffic-pressure index</span></div>
      <div class="hero-sub">This score is an illustrative model output, not a travel-time estimate or a live traffic reading.</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(min(max(int(prediction),0),100))
    st.markdown("**Interpretation**")
    if prediction < 35:
        st.success("Lower modelled pressure for the selected scenario.")
    elif prediction < 70:
        st.warning("Moderate modelled pressure. Consider comparing alternative scenarios.")
    else:
        st.error("Higher modelled pressure in this scenario. Validate against real traffic data before making operational decisions.")

elif page == "Model Evaluation":
    st.markdown("## Model evaluation")
    st.markdown('<div class="small-muted">Holdout evaluation on a synthetic dataset. Metrics demonstrate model evaluation practices; they do not measure real-world traffic forecasting accuracy.</div>', unsafe_allow_html=True)
    m1,m2,m3 = st.columns(3)
    m1.metric("MAE", f'{evaluation["mae"]:.2f}')
    m2.metric("RMSE", f'{evaluation["rmse"]:.2f}')
    m3.metric("R²", f'{evaluation["r2"]:.3f}')
    st.markdown("### Actual vs predicted")
    from sklearn.model_selection import train_test_split
    from src.model import FEATURE_COLUMNS, TARGET_COLUMN
    X = historical[FEATURE_COLUMNS]
    y = historical[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    eval_df = pd.DataFrame({"Actual index": y_test.values, "Predicted index": y_pred})
    fig = px.scatter(eval_df, x="Actual index", y="Predicted index", opacity=.65, title="Holdout predictions")
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dash", color="#61e6c4"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Evaluation notes")
    st.write("- Train/test split uses a fixed random seed for reproducibility.")
    st.write("- MAE and RMSE are measured in index points; lower is better.")
    st.write("- R² describes fit on the synthetic holdout sample and may not generalize to real city traffic.")
    st.write("- For a production-grade digital twin, replace simulated traffic and energy features with licensed, timestamped operational feeds and evaluate on time-based splits.")

st.divider()
st.caption("UrbanPulse AI · Portfolio prototype · Not an official city monitoring or emergency-response system.")
