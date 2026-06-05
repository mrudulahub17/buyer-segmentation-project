import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("clients.csv")

# ======================
# CLEAN DATA
# ======================
df.drop_duplicates(inplace=True)

# ======================
# DOB → AGE
# ======================
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")

df["age"] = datetime.now().year - df["date_of_birth"].dt.year

df["age"] = df["age"].fillna(df["age"].median())

# ======================
# DROP TEXT IDENTIFIERS
# ======================
df = df.drop(columns=[
    "client_id",
    "first_name",
    "last_name",
    "date_of_birth"
])

# ======================
# ENCODE CATEGORICAL
# ======================
cat_cols = [
    "client_type",
    "gender",
    "country",
    "region",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel"
]

for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# ======================
# SCALE ONLY NUMERIC COLUMNS
# (IMPORTANT FIX FOR WARNINGS)
# ======================
num_cols = ["age", "satisfaction_score"]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ======================
# KMEANS CLUSTERING
# ======================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(df)

# ======================
# CLUSTER LABELS
# ======================
cluster_map = {
    0: "Global Investors",
    1: "First-Time Buyers",
    2: "Corporate Buyers",
    3: "Luxury Investors"
}

df["segment"] = df["cluster"].map(cluster_map)

# ======================
# SAVE OUTPUT
# ======================
df.to_csv("clustered_buyers.csv", index=False)

# ======================
# OUTPUT CHECK
# ======================
print("clustering Completed ✔")
print(df["cluster"].value_counts())
print(df["segment"].value_counts())