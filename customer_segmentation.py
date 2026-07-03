import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# GENERATE MOCK DATA (Kaggle Style)
np.random.seed(42)
num_customers = 300

# Create realistic distinct user groups (e.g., low income/high spend, high income/low spend)
group1 = np.random.multivariate_normal([25, 80], [[10, 0], [0, 10]], 60) # High Spend, Low Income
group2 = np.random.multivariate_normal([85, 85], [[15, 0], [0, 15]], 60) # High Spend, High Income
group3 = np.random.multivariate_normal([55, 50], [[12, 0], [0, 12]], 60) # Standard Average Customers
group4 = np.random.multivariate_normal([80, 15], [[10, 0], [0, 10]], 60) # Low Spend, High Income
group5 = np.random.multivariate_normal([20, 20], [[8, 0],  [0, 8]],  60) # Low Spend, Low Income

all_features = np.vstack([group1, group2, group3, group4, group5])

df = pd.DataFrame(all_features, columns=['annual_income_k', 'spending_score_1_100'])
df['customer_id'] = range(1, num_customers + 1)
df['age'] = np.random.randint(18, 70, size=num_customers)

# Reorder columns to mimic raw transactional data
df = df[['customer_id', 'age', 'annual_income_k', 'spending_score_1_100']]
print("--- Unlabeled Mall Dataset Sample ---")
print(df.head(), "\n")

# PREPARE AND SCALE FEATURES
# For clustering, we select the major behavioral metrics
X = df[['annual_income_k', 'spending_score_1_100']]

# Distance-based models like K-Means heavily require feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# THE ELBOW METHOD (FINDING OPTIMAL 'K')
# We run K-Means iteratively to find where adding more groups stops adding value
wcss = [] # Within-Cluster Sum of Squares
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

print("--- Within-Cluster Sum of Squares (WCSS) per K ---")
for idx, val in enumerate(wcss, 1):
    print(f"Clusters (k)={idx}: WCSS = {val:.2f}")

# FIT FINAL K-MEANS MODEL
# Based on the data generated, the mathematical "elbow drop" happens precisely at K=5
optimal_k = 5
final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['cluster_label'] = final_kmeans.fit_predict(X_scaled)

print("\n--- Final Segmented Output ---")
print(df[['customer_id', 'annual_income_k', 'spending_score_1_100', 'cluster_label']].head(10))

# PORTFOLIO VISUALIZATION
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(optimal_k):
    cluster_data = df[df['cluster_label'] == i]
    plt.scatter(cluster_data['annual_income_k'], cluster_data['spending_score_1_100'], 
                s=50, c=colors[i], label=f'Cluster {i+1}')

plt.title('Mall Customer Persona Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

plt.show()