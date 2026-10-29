import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Sales Prediction & Analytics", layout="wide")

st.title("📊 AI-Powered Sales Analytics & Forecasting")
st.markdown("Analyze historical sales performance and forecast future revenue using machine learning.")

# Sidebar Controls
st.sidebar.header("Navigation & Settings")
app_mode = st.sidebar.selectbox("Choose View", ["Analytics Dashboard", "Sales Predictor"])

# Dummy dataset generator for demonstration if no CSV exists
@st.cache_data
def load_demo_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2026-10-29")
    base_sales = 500 + np.sin(np.linspace(0, 20, len(dates))) * 150
    noise = np.random.normal(0, 50, len(dates))
    sales = np.clip(base_sales + noise, 100, None)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'category': np.random.choice(['Electronics', 'Apparel', 'Home Goods'], size=len(dates))
    })
    return df

df = load_demo_data()

if app_mode == "Analytics Dashboard":
    st.header("Performance Analytics")
    
    # Metric cards
    total_sales = df['sales'].sum()
    avg_daily_sales = df['sales'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Recorded Sales", f"${total_sales:,.2f}")
    col2.metric("Average Daily Sales", f"${avg_daily_sales:,.2f}")
    col3.metric("Data Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    st.divider()
    
    # Sales Trend Line Chart
    st.subheader("Historical Sales Trend Over Time")
    fig = px.line(df, x='date', y='sales', title="Daily Sales Volume")
    st.plotly_chart(fig, use_container_width=True)
    
    # Category Breakdown
    st.subheader("Sales by Product Category")
    category_sales = df.groupby('category')['sales'].sum().reset_index()
    fig_pie = px.pie(category_sales, names='category', values='sales', hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

elif app_mode == "Sales Predictor":
    st.header("Future Sales Forecasting Tool")
    st.markdown("Input variables below to generate automated sales predictions.")
    
    col1, col2 = st.columns(2)
    with col1:
        month = st.slider("Month", 1, 12, 6)
        day = st.slider("Day of Month", 1, 31, 15)
        day_of_week = st.selectbox("Day of Week", options=[0,1,2,3,4,5,6], format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
    
    with col2:
        sales_lag_1 = st.number_input("Previous Day Sales ($)", min_value=0.0, value=520.0)
        sales_lag_7 = st.number_input("Sales 7 Days Ago ($)", min_value=0.0, value=490.0)
        sales_rolling_mean_7 = st.number_input("7-Day Rolling Average Sales ($)", min_value=0.0, value=505.0)
    
    is_weekend = 1 if day_of_week in [5, 6] else 0
    
    if st.button("Predict Sales", type="primary"):
        # Check if model exists, otherwise use a fallback heuristic estimator for demo
        model_path = 'models/sales_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            input_data = pd.DataFrame([[month, day, day_of_week, is_weekend, sales_lag_1, sales_lag_7, sales_rolling_mean_7]],
                                     columns=['month', 'day', 'day_of_week', 'is_weekend', 'sales_lag_1', 'sales_lag_7', 'sales_rolling_mean_7'])
            prediction = model.predict(input_data)[0]
        else:
            # Fallback estimation if model hasn't been trained on local file yet
            prediction = (sales_lag_1 * 0.4) + (sales_rolling_mean_7 * 0.6) + (50 if is_weekend else 0)
            st.warning("Using fallback heuristic estimator (train the model script with real data for machine learning predictions).")
            
        st.success(f"### Predicted Sales: ${prediction:,.2f}")