# 🏠 Real Estate Buyer Segmentation using Machine Learning

## 📌 Project Overview
This project uses Machine Learning (K-Means Clustering) to segment real estate buyers based on their demographics, behavior, and investment patterns.  
A Streamlit dashboard is built to visualize insights interactively.

---

## 🎯 Problem Statement
Real estate companies treat all buyers the same, leading to:
- Inefficient marketing
- Poor targeting
- Missed investment opportunities

This project solves this by identifying hidden buyer segments using clustering.

---

## 📊 Dataset Features
- client_id
- client_type (Individual / Corporate)
- gender
- country
- region
- date_of_birth
- acquisition_purpose (Investment / Home use)
- loan_applied (Yes/No)
- referral_channel
- satisfaction_score

---

## 🧠 Machine Learning Approach

### Steps Followed:
1. Data Cleaning (missing values, duplicates)
2. Feature Engineering (Age from DOB)
3. Encoding categorical variables
4. Feature scaling
5. K-Means Clustering
6. Cluster interpretation

---

## 📦 Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

## 📊 Cluster Segments

| Cluster | Segment Type |
|--------|-------------|
| 0 | Global Investors |
| 1 | First-Time Buyers |
| 2 | Corporate Buyers |
| 3 | Luxury Investors |

---

## 🚀 How to Run This Project

### 1. Clone repository
```bash
git clone https://github.com/your-username/repo-name.git
