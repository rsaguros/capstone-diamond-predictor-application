# main.py

from Analyzer import Analyzer
from Classifier import Classifier
from Regressor import Regressor
from Clustering import Clustering

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Used for displaying clustering axis
FEATURE_GROUPS = {
    "Size": ["carat", "x", "y", "z"],
    "Price": ["price"],
    "Proportions": ["table", "depth"],
    "Cut Quality": ["cut_Fair", "cut_Good", "cut_Ideal", "cut_Premium", "cut_Very Good"],
    "Color": ["color_D", "color_E", "color_F", "color_G", "color_H", "color_I", "color_J"],
    "Clarity": ["clarity_I1", "clarity_IF", "clarity_SI1", "clarity_SI2", "clarity_VS1", "clarity_VS2", "clarity_VVS1", "clarity_VVS2"]
}

# ============================================================
# 1. Classification Scenario
# ============================================================
def run_classification_scenario():
    print("\n==================== CLASSIFICATION SCENARIO ====================\n")

    analyzer = Analyzer()
    analyzer.read_dataset("diamonds.csv")

    # Preprocess
    analyzer.drop_missing_data()
    analyzer.encode_features(["color", "clarity"])
    label_encoder = analyzer.encode_label("cut")
    analyzer.shuffle()

    data = analyzer.retrieve_data()

    # Features and target
    X = data.drop(columns=["cut"])
    y = data["cut"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = Classifier()

    # Example: KNN classification
    print("\n--- Running KNN Classifier ---")
    clf.knn_classifier(X_train, y_train, X_test, y_test)

    # Evaluate
    y_pred = clf.predict(X_test)
    clf.plot_confusionMatrix(y_test, y_pred, label_encoder.classes_)

# ============================================================
# 2. Regression Scenario
# ============================================================
def run_regression_scenario():
    print("\n==================== REGRESSION SCENARIO ====================\n")

    analyzer = Analyzer()
    analyzer.read_dataset("diamonds.csv")

    # Preprocess
    analyzer.drop_missing_data()
    analyzer.encode_features(["cut", "color", "clarity"])
    analyzer.shuffle()

    data = analyzer.retrieve_data()

    # Features and target
    X = data.drop(columns=["price"])
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    reg = Regressor()

    # Example: Random Forest Regression
    print("\n--- Running Random Forest Regressor ---")
    scores = reg.random_forest(X_train, y_train, X_test, y_test)
    print("Scores:", scores)

    # Evaluate with unified score function
    print("R2 Score:", reg.score(X_test, y_test, "r2"))
    print("MSE:", reg.score(X_test, y_test, "mse"))
    print("MAE:", reg.score(X_test, y_test, "mae"))
    print("RMSE:", reg.score(X_test, y_test, "rmse"))
    
    # ------------------------------------------------------------
    # 2.1 Single Diamond Price Prediction Demo
    # ------------------------------------------------------------
    print("\n--- Single Diamond Price Prediction Demo ---")

    # Take one example diamond from the test set
    example_features = X_test.iloc[0:1]
    actual_price = y_test.iloc[0]

    predicted_price = reg.predict(example_features)[0]

    print("Example diamond features:")
    print(example_features)

    print(f"\nPredicted Price: ${predicted_price:,.2f}")
    print(f"Actual Price:    ${actual_price:,.2f}")

# ============================================================
# 3. Clustering Scenario
# ============================================================
def run_clustering_scenario():
    print("\n==================== CLUSTERING SCENARIO ====================\n")

    analyzer = Analyzer()
    analyzer.read_dataset("diamonds.csv")
    
    # Preprocess
    analyzer.drop_missing_data()
    analyzer.encode_features(["cut", "color", "clarity"])
    analyzer.shuffle()

    data = analyzer.retrieve_data()

    # Use only numeric features for clustering and meaningful data
    X = data.select_dtypes(include=["number", "bool"])

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clusterer = Clustering()

    # Example: K-Means with elbow method
    print("\n--- Running K-Means Clustering ---")
    inertia_list = clusterer.kmeans(X)
    print("Inertia values:", inertia_list)

    # Example: Agglomerative Clustering
    print("\n--- Running Agglomerative Clustering ---")
    clusterer.agglomerative(X, n_clusters=4)

    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=["PC1", "PC2"]
    )
    axis_labels = interpret_pca_components(loadings, FEATURE_GROUPS)

    visualize_clusters(X_scaled, clusterer.labels_, "Agglomerative Clustering", data, axis_labels["PC1"], axis_labels["PC2"])

    # Example: Mean-Shift Clustering
    # print("\n--- Running Mean-Shift Clustering ---")
    # clusterer.mean_shift(X)
    # visualize_clusters(X, clusterer.labels_, "Mean-Shift Clustering", data, axis_labels["PC1"], axis_labels["PC2"])

    # Example for demo: Mean-Shift Clustering with sampling
    print("\n--- Running Mean-Shift Clustering (Sampled) ---")

    # Sample 10% of the dataset for faster Mean-Shift
    analyzer.sample(0.1)
    sampled_data = analyzer.retrieve_data()

    # Extract numeric features
    X_sampled = sampled_data.select_dtypes(include=["number", "bool"])

    # Scale
    scaler = StandardScaler()
    X_sampled_scaled = scaler.fit_transform(X_sampled)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_sampled_pca = pca.fit_transform(X_sampled_scaled)

    # Run Mean-Shift on the scaled sampled data
    clusterer.mean_shift(X_sampled_scaled)

    # Visualize clusters using PCA projection
    visualize_clusters(
        X_sampled_pca,
        clusterer.labels_,
        "Mean-Shift Clustering (Sampled)",
        sampled_data,
        axis_labels["PC1"],
        axis_labels["PC2"]
    )

# ============================================================
# Clustering Visualization Functions
# ============================================================
def summarize_clusters(df, labels):
    df = df.copy()
    df["cluster"] = labels

    summary = df.groupby("cluster").mean(numeric_only=True)
    return summary

def name_cluster(row):
    # Size category
    if row["carat"] > 1.0:
        size = "Large"
    elif row["carat"] > 0.7:
        size = "Mid-size"
    elif row["carat"] > 0.45:
        size = "Small"
    else:
        size = "Tiny"

    # Price category
    if row["price"] > 6000:
        price = "Expensive"
    elif row["price"] > 3000:
        price = "Mid-price"
    else:
        price = "Cheap"

    # Clarity category (using one-hot means)
    clarity_cols = [c for c in row.index if c.startswith("clarity_")]
    clarity = max(clarity_cols, key=lambda c: row[c]).replace("clarity_", "")

    # Cut category
    cut_cols = [c for c in row.index if c.startswith("cut_")]
    cut = max(cut_cols, key=lambda c: row[c]).replace("cut_", "")

    # Build readable name
    return f"{size} {price} ({cut}, {clarity})"

def generate_cluster_names(df, labels):
    summary = summarize_clusters(df, labels)
    names = {}

    for cluster_id, row in summary.iterrows():
        names[cluster_id] = name_cluster(row)

    return names

def interpret_pca_components(loadings, feature_groups, top_n=2):
    labels = {}
    
    for pc in loadings.columns:
        scores = {}
        for concept, features in feature_groups.items():
            present = [f for f in features if f in loadings.index]
            if present:
                scores[concept] = loadings.loc[present, pc].abs().sum()
        
        # Pick the top N concepts
        top_concepts = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        label = " + ".join([c[0] for c in top_concepts])
        labels[pc] = label
    
    return labels

def visualize_clusters(X, labels, title, df_original=None, x_label="PCA Component 1", y_label="PCA Component 2"):
    # PCA
    if X.shape[1] == 2:
        X_2d = X if isinstance(X, np.ndarray) else X.values
    else:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

    # Generate names if df_original is provided
    if df_original is not None:
        cluster_names = generate_cluster_names(df_original, labels)
    else:
        cluster_names = {l: f"Cluster {l}" for l in set(labels)}

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis", s=10)

    # Legend
    unique_labels = sorted(set(labels))
    legend_handles = scatter.legend_elements()[0]
    legend_labels = [cluster_names[l] for l in unique_labels]
    plt.legend(legend_handles, legend_labels, title="Clusters")

    # # Centroid labels
    # for cluster_id in unique_labels:
    #     cluster_points = X_2d[labels == cluster_id]
    #     cx = cluster_points[:, 0].mean()
    #     cy = cluster_points[:, 1].mean()

    #     plt.text(
    #         cx, cy,
    #         cluster_names[cluster_id],
    #         fontsize=10, weight="bold",
    #         ha="center", va="center",
    #         bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
    #     )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


if __name__ == "__main__":
    run_classification_scenario()
    input("Continue...")
    run_regression_scenario()
    input("Continue...")
    run_clustering_scenario()

    print("\n==================== PROGRAM FINISHED ====================\n")
