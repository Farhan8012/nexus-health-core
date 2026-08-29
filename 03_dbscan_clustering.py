import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
# Importing our modular data pipeline from Day 1
from 01_patient_clustering import ingest_and_clean_data, scale_patient_features

sns.set_theme(style="darkgrid")

def apply_dbscan(df_scaled, eps=1.5, min_samples=5):
    """Applies Density-Based Clustering to find organic groups and isolate noise."""
    print(f"--- 1. Running DBSCAN (eps={eps}, min_samples={min_samples}) ---")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(df_scaled)
    
    # Calculate how many clusters were found and how much noise exists
    # We subtract 1 from the cluster count if noise (-1) is present
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"DBSCAN automatically found {n_clusters} distinct health profiles.")
    print(f"DBSCAN isolated {n_noise} patients as pure noise (critical anomalies).")
    
    return cluster_labels

def visualize_dbscan(df_scaled, dbscan_labels):
    """Compresses data to 2D using PCA and visualizes the DBSCAN organic clusters."""
    print("--- 2. Visualizing DBSCAN Results ---")
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    
    plot_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    plot_df['Cluster'] = dbscan_labels
    
    plt.figure(figsize=(10, 7))
    
    # Create a color palette: one color for each valid cluster
    unique_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    palette = sns.color_palette("Set2", unique_clusters)
    
    # Insert black at the beginning of the palette specifically for the -1 (noise) label
    if -1 in dbscan_labels:
        palette.insert(0, (0.0, 0.0, 0.0)) 
    
    sns.scatterplot(
        x='PCA1', 
        y='PCA2',
        hue='Cluster',
        palette=palette,
        data=plot_df,
        legend='full',
        alpha=0.8,
        s=50 # dot size
    )
    plt.title('DBSCAN Clustering (Black dots are -1 Anomalies)')
    plt.xlabel('Principal Component 1 (Main Health Variance)')
    plt.ylabel('Principal Component 2 (Secondary Health Variance)')
    plt.show()

if __name__ == "__main__":
    # 1. Pull clean, scaled data using our existing pipeline
    raw_data = ingest_and_clean_data()
    scaled_data = scale_patient_features(raw_data)
    
    # 2. Run DBSCAN
    dbscan_labels = apply_dbscan(scaled_data, eps=1.5, min_samples=5)
    
    # 3. Attach the new labels to our original data
    raw_data['DBSCAN_Cluster'] = dbscan_labels
    
    # 4. Visualize
    visualize_dbscan(scaled_data, dbscan_labels)