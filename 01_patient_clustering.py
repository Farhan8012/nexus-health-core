import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Set plotting style
sns.set_theme(style="whitegrid")

def ingest_and_clean_data(filepath="patient_vitals.csv"):
    """Loads raw patient data and handles missing values."""
    print("--- 1. Ingesting Data ---")
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded dataset with shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure the dataset is in the folder.")
        # Creating dummy data so the script still runs if you don't have the CSV yet
        print("Generating synthetic patient data for testing...")
        np.random.seed(42)
        df = pd.DataFrame(np.random.randint(60, 200, size=(1000, 5)), 
                          columns=['HeartRate', 'BloodPressure', 'Glucose', 'BMI', 'Cholesterol'])

    # Impute missing values with the median of each column
    imputer = SimpleImputer(strategy='median')
    df_cleaned = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    print("Missing values handled. Data is clean.\n")
    return df_cleaned

def scale_patient_features(df):
    """Standardizes features so they have a mean of 0 and variance of 1."""
    print("--- 2. Scaling Features ---")
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    
    # Convert back to DataFrame for easier handling
    df_scaled = pd.DataFrame(scaled_array, columns=df.columns)
    print("Features scaled successfully. (Crucial for distance-based ML like K-Means)\n")
    return df_scaled

def build_patient_clusters(df_scaled, n_clusters=4):
    """Applies K-Means clustering to identify hidden patient risk groups."""
    print(f"--- 3. Running K-Means Clustering (k={n_clusters}) ---")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # The ML model learns the patterns and assigns a cluster label to each patient
    cluster_labels = kmeans.fit_predict(df_scaled)
    print("Clustering complete.\n")
    return cluster_labels, kmeans

def visualize_clinical_clusters(df_scaled, cluster_labels):
    """Reduces multidimensional data to 2D using PCA and plots the clusters."""
    print("--- 4. Visualizing the Results ---")
    # We have 5+ features, but our screens are 2D. PCA compresses the data 
    # while keeping the most important variance so we can graph it.
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    
    # Create a new DataFrame for plotting
    plot_df = pd.DataFrame(data=pca_result, columns=['PCA_Component_1', 'PCA_Component_2'])
    plot_df['Cluster'] = cluster_labels
    
    # Plotting
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='PCA_Component_1', 
        y='PCA_Component_2',
        hue='Cluster',
        palette='viridis',
        data=plot_df,
        legend='full',
        alpha=0.7
    )
    plt.title('Patient Health Clusters (PCA Reduced)')
    plt.xlabel('Principal Component 1 (Main Health Variance)')
    plt.ylabel('Principal Component 2 (Secondary Health Variance)')
    plt.show()

if __name__ == "__main__":
    # The Pipeline Execution
    # 1. Load
    raw_data = ingest_and_clean_data()
    
    # 2. Scale
    scaled_data = scale_patient_features(raw_data)
    
    # 3. Cluster
    # We choose 4 clusters (e.g., Healthy, At-Risk, Chronic, Critical)
    labels, model = build_patient_clusters(scaled_data, n_clusters=4)
    
    # Add the labels back to our original readable data to see who is in which group
    raw_data['Risk_Cluster'] = labels
    print("Sample of clustered patients:")
    print(raw_data.head())
    
    # 4. Visualize
    visualize_clinical_clusters(scaled_data, labels)