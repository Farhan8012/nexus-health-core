import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from 01_patient_clustering import (
    ingest_and_clean_data, 
    scale_patient_features, 
    build_patient_clusters, 
    detect_critical_anomalies
)

def generate_cluster_profiles(raw_df):
    """Computes mean vitals per cluster to understand what each cluster represents."""
    print("--- 1. Computing Clinical Cluster Profiles ---")
    
    # Exclude anomaly flag when computing cluster baseline means
    feature_cols = [col for col in raw_df.columns if col not in ['Risk_Cluster', 'Is_Anomaly']]
    
    # Group by the cluster label and calculate average metrics
    profile_df = raw_df.groupby('Risk_Cluster')[feature_cols].mean().round(2)
    print("Cluster Mean Vitals:")
    print(profile_df)
    return profile_df

def map_clinical_risk_labels(profile_df, raw_df):
    """Automatically maps cluster IDs to clinical risk tiers based on severity."""
    print("\n--- 2. Mapping Clinical Tiers ---")
    
    # Rank clusters based on a composite severity metric (average of scaled means)
    scaler = MinMaxScaler()
    scaled_means = scaler.fit_transform(profile_df)
    composite_severity = scaled_means.mean(axis=1)
    
    # Sort cluster IDs from lowest severity to highest severity
    ranked_clusters = np.argsort(composite_severity)
    
    tier_names = ['Low Risk / Baseline', 'Moderate Monitoring', 'High Chronic Risk', 'Critical Immediate Attention']
    
    # If the number of clusters differs, adjust tier assignment
    cluster_to_label = {}
    for rank, cluster_id in enumerate(ranked_clusters):
        tier_index = min(rank, len(tier_names) - 1)
        cluster_to_label[cluster_id] = tier_names[tier_index]
        
    print("Cluster to Risk Category Mapping:")
    for cid, label in cluster_to_label.items():
        print(f"  Cluster {cid} -> {label}")
        
    raw_df['Clinical_Risk_Tier'] = raw_df['Risk_Cluster'].map(cluster_to_label)
    return raw_df, cluster_to_label

def calculate_patient_risk_index(df_scaled, raw_df):
    """Calculates a continuous Patient Risk Index (0-100) using Euclidean distance to ideal health."""
    print("\n--- 3. Calculating Continuous Patient Risk Index (PRI) ---")
    
    # Assume ideal healthy baseline is the minimum scaled value for risk metrics
    # Calculate distance from minimum (lower is healthier, higher is at-risk)
    distances = np.linalg.norm(df_scaled.values, axis=1)
    
    # Normalize distances to a 0 - 100 scale
    scaler = MinMaxScaler(feature_range=(0, 100))
    pri_scores = scaler.fit_transform(distances.reshape(-1, 1)).flatten().round(1)
    
    raw_df['Risk_Index_Score'] = pri_scores
    return raw_df

def visualize_risk_distributions(raw_df):
    """Visualizes the distribution of Risk Index Scores across clinical tiers."""
    print("\n--- 4. Visualizing Risk Distributions ---")
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='Clinical_Risk_Tier', 
        y='Risk_Index_Score', 
        data=raw_df, 
        palette='Set2'
    )
    plt.title('Patient Risk Index (PRI) Distribution Across Clinical Tiers')
    plt.xlabel('Assigned Clinical Risk Category')
    plt.ylabel('Patient Risk Index (0 - 100)')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Run pipeline from Days 1 & 2
    raw_data = ingest_and_clean_data()
    scaled_data = scale_patient_features(raw_data)
    labels, _ = build_patient_clusters(scaled_data, n_clusters=4)
    raw_data['Risk_Cluster'] = labels
    raw_data = detect_critical_anomalies(scaled_data, raw_data)
    
    # 2. Day 3: Profile, Label, and Score
    profiles = generate_cluster_profiles(raw_data)
    labeled_data, mapping = map_clinical_risk_labels(profiles, raw_data)
    final_patient_data = calculate_patient_risk_index(scaled_data, labeled_data)
    
    print("\nSample Triage Output:")
    print(final_patient_data[['Risk_Cluster', 'Clinical_Risk_Tier', 'Risk_Index_Score', 'Is_Anomaly']].head(10))
    
    # Save the processed data for downstream supervised learning & RAG modules
    final_patient_data.to_csv("processed_patient_triage.csv", index=False)
    print("\nSaved processed triage dataset to 'processed_patient_triage.csv'")
    
    # 3. Visualize
    visualize_risk_distributions(final_patient_data)