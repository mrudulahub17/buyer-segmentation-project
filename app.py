import pandas as pd
import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Real Estate Buyer Segmentation", layout="wide")

st.title("🏠 Real Estate Buyer Segmentation Dashboard")

# =========================
# LOAD DATA (IMPORTANT FIX)
# =========================
# Original readable dataset
df_original = pd.read_csv("clients.csv")

# Clustered dataset (ML output)
df_cluster = pd.read_csv("clustered_buyers.csv")

# Merge cluster + segment into original data
df_original["cluster"] = df_cluster["cluster"]
df_original["segment"] = df_cluster["segment"]

df = df_original

# =========================
# DEBUG (optional)
# =========================
st.write("Dataset Columns:", df.columns)

# =========================
# KPI METRICS
# =========================
st.subheader("📊 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("Total Buyers", len(df))
col2.metric("Avg Satisfaction Score", round(df["satisfaction_score"].mean(), 2))
col3.metric("Total Countries", df["country"].nunique())

# =========================
# DATA PREVIEW
# =========================
st.subheader("📋 Dataset Preview")
st.dataframe(df.head())

# =========================
# CLUSTER DISTRIBUTION
# =========================
st.subheader("📊 Cluster Distribution")
st.bar_chart(df["cluster"].value_counts())

# =========================
# SEGMENT DISTRIBUTION
# =========================
st.subheader("🎯 Buyer Segments")
st.bar_chart(df["segment"].value_counts())

# =========================
# COUNTRY ANALYSIS (FIXED - READABLE)
# =========================
st.subheader("🌍 Buyers by Country")
st.bar_chart(df["country"].value_counts())

# =========================
# ACQUISITION PURPOSE
# =========================
st.subheader("📌 Acquisition Purpose")
st.bar_chart(df["acquisition_purpose"].value_counts())

# =========================
# LOAN ANALYSIS
# =========================
st.subheader("💰 Loan Applied")
st.bar_chart(df["loan_applied"].value_counts())

# =========================
# FILTER SECTION
# =========================
st.subheader("🔎 Filter Data by Country")

country = st.selectbox("Select Country", df["country"].unique())
filtered_df = df[df["country"] == country]

st.dataframe(filtered_df)

# =========================
# BUSINESS INSIGHTS
# =========================
st.subheader("📌 Business Insights")

st.write("Most Common Buyer Segment:", df["segment"].value_counts().idxmax())
st.write("Top Country:", df["country"].value_counts().idxmax())
st.write("Most Used Referral Channel:", df["referral_channel"].value_counts().idxmax())