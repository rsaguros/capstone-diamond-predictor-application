# Clustering.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from sklearn.decomposition import PCA

class Clustering:
    def __init__(self):
        """
        Clustering class supporting:
        - KMeans
        - Agglomerative Hierarchical Clustering
        - Mean-Shift Clustering
        """
        self.model = None
        self.labels_ = None
        self.is_trained = False

    # ============================================================
    # 1. K-Means Clustering
    # ============================================================
    def kmeans(self, X, k_values=range(2, 11)):
        """
        Performs K-Means clustering with multiple K values.
        Prints the optimal K using the elbow method.
        Returns the list of inertia values.
        """
        inertia_list = []

        print("Evaluating K-Means for different K values...")

        for k in k_values:
            km = KMeans(n_clusters=k)
            km.fit(X)
            inertia_list.append(km.inertia_)
            print(f"K={k}, Inertia={km.inertia_:.4f}")

        # Plot elbow curve
        plt.figure(figsize=(8, 5))
        plt.plot(list(k_values), inertia_list, marker="o")
        plt.title("Elbow Curve for K-Means")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Inertia")
        plt.grid(True)
        plt.show()

        # Determine optimal K (elbow = largest drop)
        drops = np.diff(inertia_list)
        optimal_index = np.argmin(drops) + 1
        optimal_k = list(k_values)[optimal_index]

        print(f"Optimal K (Elbow Method): {optimal_k}")

        # Train final model
        self.model = KMeans(n_clusters=optimal_k)
        self.labels_ = self.model.fit_predict(X)
        self.is_trained = True

        return inertia_list

    def get_inertia(self):
        """
        Returns the inertia_ of the trained K-Means model.
        """
        self._check_trained()
        if hasattr(self.model, "inertia_"):
            return self.model.inertia_
        else:
            raise ValueError("Current model does not have inertia_ (only K-Means supports this).")

    # ============================================================
    # 2. Agglomerative Hierarchical Clustering
    # ============================================================
    def agglomerative(self, X, n_clusters=3, linkage="ward"):
        """
        Performs Agglomerative Hierarchical Clustering.
        """
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels_ = self.model.fit_predict(X)
        self.is_trained = True

        print(f"Agglomerative Clustering completed with {n_clusters} clusters.")

    # ============================================================
    # 3. Mean-Shift Clustering
    # ============================================================
    def mean_shift(self, X):
        """
        Performs Mean-Shift clustering.
        """
        self.model = MeanShift()
        self.labels_ = self.model.fit_predict(X)
        self.is_trained = True

        print("Mean-Shift clustering completed.")
       
    # ============================================================
    # Unified API
    # ============================================================
    def fit(self, X):
        """
        Fits the current clustering model to the dataset.
        """
        self._check_model_initialized()

        if hasattr(self.model, "fit_predict"):
            self.labels_ = self.model.fit_predict(X)
        else:
            self.model.fit(X)
            self.labels_ = self.model.labels_

        self.is_trained = True
        print("Clustering model trained.")

    def predict(self, X):
        """
        Predicts cluster labels for new data.
        Only supported by models with a predict() method.
        """
        self._check_trained()

        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        else:
            raise ValueError("This clustering model does not support prediction on new data.")

    # ============================================================
    # Internal Helpers
    # ============================================================
    def _check_model_initialized(self):
        if self.model is None:
            raise ValueError("No clustering model initialized. Call a clustering method first.")

    def _check_trained(self):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() or a clustering method first.")
