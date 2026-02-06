import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS

# ---------------------------------------------------------
# 1. Data Loading and Cleaning
# ---------------------------------------------------------
data_path = 'CSVs/'
df = pd.read_csv(data_path+'similarity_lion_v3.csv')

# Clean whitespace
df['Unnamed: 1'] = df['Unnamed: 1'].str.strip()
df['cluster_name'] = df['cluster_name'].str.strip()

# Set index
df = df.set_index('Unnamed: 1')
labels = df['cluster_name']

# Extract and clean similarity matrix
similarity_matrix = df.drop(columns=['cluster_name'])
similarity_matrix.columns = similarity_matrix.columns.str.strip()

# Handle numeric conversion and potential NaNs
similarity_matrix = similarity_matrix.apply(pd.to_numeric, errors='coerce')
similarity_matrix = similarity_matrix.fillna(similarity_matrix.T)

# Ensure square matrix
common_ids = similarity_matrix.index.intersection(similarity_matrix.columns)
similarity_matrix = similarity_matrix.loc[common_ids, common_ids]
labels = labels.loc[common_ids]

# ---------------------------------------------------------
# 2. Dendrogram (Dynamic Coloring)
# ---------------------------------------------------------
distance_matrix = 1 - similarity_matrix
distance_matrix = (distance_matrix + distance_matrix.T) / 2
Z = linkage(distance_matrix, 'ward')

# --- Logic to find Optimal Color Threshold ---
# We want to color the clusters distinctively.
# 1. Find the number of unique labels (k)
k = len(labels.unique())
print(f"Number of unique clusters: {k}")

# 2. Calculate threshold
# Z[:, 2] contains merge distances. Sorted ascendingly.
# Z[-1] is the last merge (2 clusters -> 1).
# Z[-(k-1)] is the merge that creates k-1 clusters.
# We need to cut the tree *before* this merge happens, but *after* the merge that created k clusters.
if k > 1:
    # Threshold = average of height where k clusters form and where k-1 clusters form
    t = (Z[-(k-1), 2] + Z[-k, 2]) / 2
else:
    t = 0

plt.figure(figsize=(12, 8))
dendrogram(Z, labels=similarity_matrix.index, leaf_rotation=90, color_threshold=t)
#plt.title('Dendrogram')
plt.xlabel('Species')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('vis_output/dendrogram_lion.png')
plt.show()

# ---------------------------------------------------------
# 3. MDS Clustering Visualization
# ---------------------------------------------------------
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(distance_matrix)

plot_df = pd.DataFrame(coords, columns=['MDS1', 'MDS2'], index=similarity_matrix.index)
plot_df['cluster_name'] = labels

plt.figure(figsize=(12, 10))
sns.scatterplot(
    data=plot_df, x='MDS1', y='MDS2',
    hue='cluster_name', style='cluster_name',
    s=100, palette='Set1'  # 'Set1' provides strong, distinct colors
)

for i in range(plot_df.shape[0]):
    plt.text(
        plot_df.iloc[i]['MDS1'] + 0.005,
        plot_df.iloc[i]['MDS2'],
        plot_df.index[i],
        fontsize=9
    )

#plt.title('Clustering Visualization (MDS) by Individual')
plt.grid(True)
plt.tight_layout()
plt.savefig('vis_output/clustering_lion.png')
plt.show()

# ---------------------------------------------------------
# 4. Similarity Matrix (Annotated)
# ---------------------------------------------------------
sorted_indices = labels.sort_values().index
sorted_similarity = similarity_matrix.loc[sorted_indices, sorted_indices]

plt.figure(figsize=(20, 16))
sns.heatmap(
    sorted_similarity,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 6},
    cmap='coolwarm',
    xticklabels=True,
    yticklabels=True
)
#plt.title('Similarity Matrix (Sorted by Individual)')
plt.tight_layout()
plt.savefig('vis_output/similarity_lion.png')
plt.show()