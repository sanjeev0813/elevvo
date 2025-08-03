import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")

df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

X = df[['annual_income_(k$)', 'spending_score_(1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='annual_income_(k$)', y='spending_score_(1-100)', hue='kmeans_cluster', palette='Set2')
plt.title('Customer Segments (K-Means)')
plt.show()

cluster_summary = df.groupby('kmeans_cluster')[['annual_income_(k$)', 'spending_score_(1-100)']].mean()
print("Average income and spending per KMeans cluster:\n", cluster_summary)


db = DBSCAN(eps=0.5, min_samples=5)
df['dbscan_cluster'] = db.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['dbscan_cluster'], palette='tab10')
plt.title('Customer Segments (DBSCAN)')
plt.xlabel('Scaled Annual Income')
plt.ylabel('Scaled Spending Score')
plt.show()

print("Silhouette Score (KMeans, k=5):", silhouette_score(X_scaled, df['kmeans_cluster']))
