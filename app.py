import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

#pdf generation import 
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import units
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import A4
import tempfile
import os

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="AI Road Accident Risk Dashboard",
                   layout="wide")

# ---------------------------------------------------
# LOAD DATA & MODEL
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/nepal_road_accidents.csv")

@st.cache_resource
def load_model():
    model = joblib.load("model/severity_model.pkl")
    encoders = joblib.load("model/encoders.pkl")
    return model, encoders

df = load_data()
model, encoders = load_model()

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.markdown("""
<div style='background-color:#1F4E79;
            color:white;
            padding:12px;
            border-radius:8px;
            text-align:center;
            font-size:20px;
            font-weight:bold;'>
    🚦 Road Risk Filters
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.header("🔎 Filter Data")

province_filter = st.sidebar.multiselect(
    "Province",
    options=df["Province"].unique(),
    default=df["Province"].unique()
)

year_filter = st.sidebar.multiselect(
    "Year",
    options=df["Year"].unique(),
    default=df["Year"].unique()
)

severity_filter = st.sidebar.multiselect(
    "Severity",
    options=df["Severity"].unique(),
    default=df["Severity"].unique()
)

filtered_df = df[
    (df["Province"].isin(province_filter)) &
    (df["Year"].isin(year_filter)) &
    (df["Severity"].isin(severity_filter))
]

# ---------------------------------------------------
# SECTION 1 – OVERALL ANALYTICS
# ---------------------------------------------------

st.markdown("""
<div style='background-color:#E8F4FD;padding:20px;border-radius:10px'>
""", unsafe_allow_html=True)

st.title("🚦 Road Accident Overall Analytics")

# ---------------- KPIs ----------------
total_accidents = len(filtered_df)
total_deaths = filtered_df["Deaths"].sum()
total_injuries = filtered_df["Injuries"].sum()
fatality_rate = (total_deaths / total_accidents) * 100 if total_accidents > 0 else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Accidents", total_accidents)
col2.metric("Total Deaths", total_deaths)
col3.metric("Total Injuries", total_injuries)
col4.metric("Fatality Rate (%)", round(fatality_rate, 2))

st.divider()


#pdf generation code


def generate_pdf_report(df, total_accidents, total_deaths, total_injuries, fatality_rate):

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf = SimpleDocTemplate(temp_file.name, pagesize=A4)

    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("<b>AI Road Accident Risk Dashboard Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    # KPI Section
    elements.append(Paragraph("<b>Key Performance Indicators</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    kpi_data = [
        ["Total Accidents", total_accidents],
        ["Total Deaths", total_deaths],
        ["Total Injuries", total_injuries],
        ["Fatality Rate (%)", round(fatality_rate, 2)]
    ]

    table = Table(kpi_data, colWidths=[200, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10)
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    # Province Summary
    elements.append(Paragraph("<b>Province Risk Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    province_summary = df.groupby("Province").size().reset_index(name="Accidents")

    data = [["Province", "Accidents"]] + province_summary.values.tolist()

    province_table = Table(data, colWidths=[200, 150])
    province_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 9)
    ]))

    elements.append(province_table)

    pdf.build(elements)

    return temp_file.name

#pdf download button 

st.sidebar.subheader("📄 Download Dashboard Report")

if st.sidebar.button("Generate PDF Report"):

    pdf_path = generate_pdf_report(
        filtered_df,
        total_accidents,
        total_deaths,
        total_injuries,
        fatality_rate
    )

    with open(pdf_path, "rb") as file:
        st.sidebar.download_button(
            label="⬇ Download Report",
            data=file,
            file_name="road_accident_report.pdf",
            mime="application/pdf"
        )

# ---------------- Plots ----------------

# Yearly Trend
yearly = filtered_df.groupby("Year").size().reset_index(name="Accidents")
fig1 = px.line(yearly, x="Year", y="Accidents",
               title="Yearly Accident Trend", markers=True)
st.plotly_chart(fig1, use_container_width=True)

# Province Distribution
province_data = filtered_df.groupby("Province").size().reset_index(name="Count")
fig2 = px.bar(province_data, x="Province", y="Count",
              title="Accidents by Province")
st.plotly_chart(fig2, use_container_width=True)

# Vehicle Type Distribution
fig3 = px.pie(filtered_df,
              names="Vehicle_Type",
              title="Vehicle Type Distribution")
st.plotly_chart(fig3, use_container_width=True)

# Weather Impact
fig4 = px.bar(filtered_df,
              x="Weather",
              title="Weather Condition Impact")
st.plotly_chart(fig4, use_container_width=True)

# Risk Score
filtered_df["Risk_Score"] = (
    filtered_df["Deaths"] * 3 +
    filtered_df["Injuries"] * 1.5
)

risk_by_province = filtered_df.groupby("Province")["Risk_Score"].sum().reset_index()

fig5 = px.bar(risk_by_province,
              x="Province",
              y="Risk_Score",
              title="AI-Based Road Accident Risk Score by Province")

st.plotly_chart(fig5, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# SECTION 2 – AI PREDICTION & FORECASTING
# ---------------------------------------------------

st.markdown("""
<div style='background-color:#FFF3E6;padding:20px;border-radius:10px'>
""", unsafe_allow_html=True)

st.title("🤖 AI Prediction & Forecasting Section")

# ---------------- Severity Probability ----------------
st.subheader("🔮 Severity Probability Prediction")

input_data = {}

for col in df.columns:
    if col not in ["Severity", "Accident_ID", "Accident_Date"]:
        if df[col].dtype == "object":
            input_data[col] = st.selectbox(col, df[col].unique())
        else:
            input_data[col] = st.number_input(col, value=int(df[col].mean()))

if st.button("Predict Probability"):

    input_df = pd.DataFrame([input_data])

    for col in input_df.columns:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])

    probabilities = model.predict_proba(input_df)[0]
    classes = encoders["Severity"].classes_

    prob_df = pd.DataFrame({
        "Severity": classes,
        "Probability (%)": probabilities * 100
    })

    fig_prob = px.bar(prob_df,
                      x="Severity",
                      y="Probability (%)",
                      title="Predicted Severity Probability")

    st.plotly_chart(fig_prob, use_container_width=True)

# ---------------- Forecasting ----------------
st.subheader("📈 6-Month Accident Forecast")

monthly = filtered_df.groupby(["Year", "Month"]).size().reset_index(name="count")
monthly["ds"] = pd.to_datetime(
    monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01"
)
monthly = monthly.rename(columns={"count": "y"})[["ds", "y"]]

if len(monthly) > 12:

    model_prophet = Prophet()
    model_prophet.fit(monthly)

    future = model_prophet.make_future_dataframe(periods=6, freq='M')
    forecast = model_prophet.predict(future)

    fig_forecast = go.Figure()

    fig_forecast.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode='lines',
        name='Prediction'
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode='lines',
        name='Upper Bound'
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        fill='tonexty',
        mode='lines',
        name='Lower Bound'
    ))

    st.plotly_chart(fig_forecast, use_container_width=True)

else:
    st.info("Not enough data for forecasting.")

st.markdown("</div>", unsafe_allow_html=True)

