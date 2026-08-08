import os
import time
import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.decomposition import PCA
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Field Operations Intelligence Dashboard", layout="wide")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing Supabase credentials in environment. Please check your .env file.")
    st.stop()

@st.cache_resource
def get_supabase_client() -> Client:
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {str(e)}")
        st.stop()

supabase = get_supabase_client()

st.title("🏗️ Pipeline Operations & Field Intelligence Dashboard")
st.markdown("---")

@st.cache_data(ttl=15)
def fetch_dashboard_data():
    """Fetches data from Supabase with retry logic for network/connection drop issues."""
    retries = 3
    for attempt in range(retries):
        try:
            projects_res = supabase.table("projects").select("*").execute()
            logs_res = supabase.table("daily_logs").select("*").execute()
            expenses_res = supabase.table("expenses").select("*").execute()
            equipment_res = supabase.table("equipment_logs").select("*").execute()

            projects = pd.DataFrame(projects_res.data) if projects_res.data else pd.DataFrame()
            logs = pd.DataFrame(logs_res.data) if logs_res.data else pd.DataFrame()
            expenses = pd.DataFrame(expenses_res.data) if expenses_res.data else pd.DataFrame()
            equipment = pd.DataFrame(equipment_res.data) if equipment_res.data else pd.DataFrame()

            if projects.empty and logs.empty and expenses.empty and equipment.empty:
                st.warning(
                    "Supabase returned no rows for projects, daily_logs, expenses, or equipment_logs. "
                    "Please verify your table data and Supabase row-level security settings."
                )

            return projects, logs, expenses, equipment
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)  # Brief delay before retrying
                continue
            else:
                st.error(f"Error fetching data from Supabase: {str(e)}")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

projects_df, logs_df, expenses_df, equipment_df = fetch_dashboard_data()

# Metric Calculations with fallback checks
try:
    total_trenching = logs_df['trenching_meters'].sum() if not logs_df.empty and 'trenching_meters' in logs_df.columns else 0
    total_pipe = logs_df['pipe_laid_meters'].sum() if not logs_df.empty and 'pipe_laid_meters' in logs_df.columns else 0
    total_expenses = expenses_df['amount'].sum() if not expenses_df.empty and 'amount' in expenses_df.columns else 0
    active_projects = len(projects_df) if not projects_df.empty else 0
except Exception as e:
    st.warning(f"Error computing top metrics: {str(e)}")
    total_trenching, total_pipe, total_expenses, active_projects = 0, 0, 0, 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Projects", active_projects)
col2.metric("Total Trenching Laid", f"{total_trenching:,} Mtr")
col3.metric("Total Pipe Laid", f"{total_pipe:,} Mtr")
col4.metric("Total Outflow Expenses", f"₹ {total_expenses:,.2f}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Operational Metrics", "💰 Financial Outflows", "🧠 Machine PCA Anomaly Detection"])

with tab1:
    st.subheader("Daily Work Progress Trends")
    try:
        required_cols = {'log_date', 'trenching_meters', 'pipe_laid_meters', 'backfilling_meters'}
        if not logs_df.empty and required_cols.issubset(logs_df.columns):
            logs_df_copy = logs_df.copy()
            logs_df_copy['log_date'] = pd.to_datetime(logs_df_copy['log_date'])
            chart_data = logs_df_copy.groupby('log_date')[['trenching_meters', 'pipe_laid_meters', 'backfilling_meters']].sum().reset_index()
            fig = px.line(chart_data, x='log_date', y=['trenching_meters', 'pipe_laid_meters', 'backfilling_meters'],
                          title="Progress (Meters) over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily logs recorded yet or necessary columns missing.")
    except Exception as e:
        st.error(f"Failed to generate operational metrics chart: {str(e)}")

with tab2:
    st.subheader("Site Expense Breakdowns")
    try:
        if not expenses_df.empty and 'amount' in expenses_df.columns:
            col_a, col_b = st.columns(2)
            with col_a:
                if 'category' in expenses_df.columns:
                    cat_fig = px.pie(expenses_df, names='category', values='amount', title="Expenditure by Category")
                    st.plotly_chart(cat_fig, use_container_width=True)
                else:
                    st.info("'category' column missing in expense records.")
            with col_b:
                if 'payment_mode' in expenses_df.columns and 'category' in expenses_df.columns:
                    pay_fig = px.bar(expenses_df, x='payment_mode', y='amount', color='category', title="Payment Mode Distribution")
                    st.plotly_chart(pay_fig, use_container_width=True)
                else:
                    st.info("'payment_mode' or 'category' column missing in expense records.")
        else:
            st.info("No expense logs recorded yet.")
    except Exception as e:
        st.error(f"Failed to generate financial charts: {str(e)}")

with tab3:
    st.subheader("Equipment Operating Anomaly Detection (PCA)")
    try:
        required_eq_cols = {'hours_operated', 'fuel_consumed_liters'}
        if not equipment_df.empty and required_eq_cols.issubset(equipment_df.columns) and len(equipment_df) >= 3:
            eq_copy = equipment_df.copy()
            X = eq_copy[['hours_operated', 'fuel_consumed_liters']].fillna(0)
            
            pca = PCA(n_components=2)
            components = pca.fit_transform(X)
            
            eq_copy['PCA_1'] = components[:, 0]
            eq_copy['PCA_2'] = components[:, 1]
            eq_copy['Anomaly_Score'] = np.sqrt(eq_copy['PCA_1']**2 + eq_copy['PCA_2']**2)
            
            hover_cols = [col for col in ['equipment_name', 'hours_operated', 'fuel_consumed_liters'] if col in eq_copy.columns]
            
            fig_pca = px.scatter(
                eq_copy, x='PCA_1', y='PCA_2', color='Anomaly_Score',
                hover_data=hover_cols,
                title="PCA Equipment Clustering & Variance Plot"
            )
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.warning("Insufficient equipment log data available to execute PCA reduction (minimum 3 valid records required).")
    except Exception as e:
        st.error(f"Failed to run PCA anomaly detection: {str(e)}")