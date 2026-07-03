An unsupervised clustering system that processes consumer transaction dynamics to reveal clean, distinct demographic personas for target marketing.

Core Architecture -
*   ML Task: Unsupervised Clustering (No target variable labels)
*   Model Implemented: K-Means Clustering (`init='k-means++'`)
*   Feature Processing: Distance-based scaling via `StandardScaler`.

Utilizes The Elbow Method tracking the Within-Cluster Sum of Squares (WCSS) to programmatically isolate the point of diminishing returns, revealing an optimal layout of 5 distinct core consumer clusters.
