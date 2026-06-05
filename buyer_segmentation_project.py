import pandas as pd

# Load data
clients = pd.read_csv("clients.csv")
properties = pd.read_csv("properties.csv")

# Clean sale price
properties['sale_price'] = (
    properties['sale_price']
    .replace({'\$':'', ',':''}, regex=True)
    .astype(float)
)

# Merge datasets
df = pd.merge(
    clients,
    properties,
    left_on='client_id',
    right_on='client_ref',
    how='inner'
)

print("Merged Shape:", df.shape)

# Convert DOB to datetime
df['date_of_birth'] = pd.to_datetime(
    df['date_of_birth'],
    errors='coerce',
    dayfirst=False
)

current_year = pd.Timestamp.now().year

df['age'] = current_year - df['date_of_birth'].dt.year

df['age'] = df['age'].fillna(df['age'].median())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = [
    'age',
    'satisfaction_score',
    'sale_price',
    'floor_area_sqft'
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster Distribution:")
print(df['Cluster'].value_counts())

df.to_csv(
    "clustered_buyers.csv",
    index=False
)

print("\nclustered_buyers.csv saved successfully")