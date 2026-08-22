import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

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

def find_optimal_clusters(df_scaled, max_k=10):
    """Uses the Elbow Method to find the optimal number of clusters."""
    print("--- Calculating Optimal Cluster Count (Elbow Method) ---")
    wcss = [] # Within-Cluster Sum of Squares (how tight the clusters are)
    
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)
        
    # Plot the Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
    plt.title('The Elbow Method (Look for the "bend")')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.show()

def detect_critical_anomalies(df_scaled, raw_data):
    """Uses Isolation Forest to flag severe patient anomalies."""
    print("--- Running Isolation Forest for Outlier Detection ---")
    # contamination=0.05 means we assume roughly 5% of patients are severe anomalies
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    
    # Predict returns 1 for normal, -1 for anomaly
    anomaly_labels = iso_forest.fit_predict(df_scaled)
    
    # Add back to our raw data dataframe
    raw_data['Is_Anomaly'] = anomaly_labels
    
    # Filter and show just the anomalies
    anomalies = raw_data[raw_data['Is_Anomaly'] == -1]
    print(f"Detected {len(anomalies)} critical patients.")
    print(anomalies.head())
    return raw_data




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
    # 1. Load & Clean
    raw_data = ingest_and_clean_data()
    
    # 2. Scale
    scaled_data = scale_patient_features(raw_data)
    
    # --- NEW: Find the best K ---
    # The code will pause and show you a graph. Close the graph to continue.
    # Look at where the line sharply bends (the "elbow"). That is your optimal K.
    find_optimal_clusters(scaled_data, max_k=10)
    
    # 3. Cluster (Change n_clusters to whatever the elbow graph suggested, e.g., 3 or 4)
    labels, model = build_patient_clusters(scaled_data, n_clusters=4)
    raw_data['Risk_Cluster'] = labels
    
    # --- NEW: Detect Anomalies ---
    raw_data = detect_critical_anomalies(scaled_data, raw_data)
    
    # 4. Visualize
    visualize_clinical_clusters(scaled_data, labels)