# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Step 1: Load the dataset
# Sample dataset; replace with your dataset file path
data = {
    "CustomerID": [1, 2, 3, 4, 5],
    "Gender": ["Male", "Female", "Female", "Male", "Female"],
    "Age": [19, 21, 20, 23, 31],
    "Annual Income (k$)": [15, 16, 17, 18, 20],
    "Spending Score (1-100)": [39, 81, 6, 77, 40],
}

df = pd.DataFrame(data)

# Step 2: Preprocess the data
# Encode Gender (Categorical data)
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# Select features for clustering
features = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply the K-means clustering algorithm
# Adjust K_range to ensure K is <= number of samples
K_range = range(1, min(6, len(df)) + 1)  # Max K is 5 since we have 5 samples
inertia = []

# Determine optimal number of clusters using the Elbow Method
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method to Find Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Step 4: Fit K-means with the optimal number of clusters (e.g., K=2)
kmeans = KMeans(n_clusters=2, random_state=42)  # You can adjust K based on the Elbow Method
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Step 5: Visualize the clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df["Annual Income (k$)"], df["Spending Score (1-100)"],
    c=df["Cluster"], cmap="viridis", s=100, alpha=0.7
)
plt.colorbar(scatter)
plt.title("K-means Clustering of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.show()

# Step 6: Display clustered data
print("Clustered Data:\n", df)
