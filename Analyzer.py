# Analyzer.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle as sk_shuffle


class Analyzer:
    def __init__(self):
        self.data = None

    # ============================================================
    # 1. read_dataset
    # ============================================================
    def read_dataset(self, csv_file):
        """
        Reads a CSV file and stores it in the analyzer instance.
        This assumes the first column of the CSV is an index column.
        """
        self.data = pd.read_csv(csv_file, index_col=0)
        print(f"Dataset loaded: {csv_file}")
        print(f"Shape: {self.data.shape}")

    # ============================================================
    # 2. describe
    # ============================================================
    def describe(self):
        """
        Prints attribute types and basic statistical analysis.
        """
        self._check_data_loaded()

        print("\n=== Attribute Types ===")
        print(self.data.dtypes)

        print("\n=== Statistical Summary (Numeric Attributes) ===")
        print(self.data.describe())

    # ============================================================
    # 3. drop_missing_data
    # ============================================================
    def drop_missing_data(self):
        """
        Drops any sample (row) with missing values.
        """
        self._check_data_loaded()

        before = len(self.data)
        self.data = self.data.dropna()
        after = len(self.data)

        print(f"Dropped {before - after} rows containing missing values.")

    # ============================================================
    # 4. drop_columns
    # ============================================================
    def drop_columns(self, columns):
        """
        Drops a list of attribute names from the dataset.
        """
        self._check_data_loaded()

        self.data = self.data.drop(columns=columns)
        print(f"Dropped columns: {columns}")

    # ============================================================
    # 5. encode_features
    # ============================================================
    def encode_features(self, columns):
        """
        One-hot encodes a list of nominal feature columns.
        """
        self._check_data_loaded()

        self.data = pd.get_dummies(self.data, columns=columns)
        print(f"One-hot encoded feature columns: {columns}")

    # ============================================================
    # 6. encode_label
    # ============================================================
    def encode_label(self, target_column):
        """
        Label-encodes the target column for classification tasks.
        Returns the label encoder.
        """
        self._check_data_loaded()

        le = LabelEncoder()
        self.data[target_column] = le.fit_transform(self.data[target_column].astype(str))

        print(f"Label-encoded target column: {target_column}")
        return le

    # ============================================================
    # 7. shuffle
    # ============================================================
    def shuffle(self):
        """
        Shuffles the dataset rows.
        """
        self._check_data_loaded()

        self.data = sk_shuffle(self.data).reset_index(drop=True)
        print("Dataset shuffled.")

    # ============================================================
    # 8. retrieve_data
    # ============================================================
    def retrieve_data(self):
        """
        Returns the dataset stored in the analyzer instance.
        """
        self._check_data_loaded()
        return self.data

    # ============================================================
    # 9. plot_correlationMatrix
    # ============================================================
    def plot_correlationMatrix(self):
        """
        Plots an annotated correlation matrix and saves it as Correlation.png.
        """
        self._check_data_loaded()

        corr = self.data.corr(numeric_only=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig("Correlation.png")
        plt.show()

        print("Correlation matrix saved as Correlation.png")

    # ============================================================
    # 10. plot_pairPlot
    # ============================================================
    def plot_pairPlot(self):
        """
        Plots a pair-plot of all attributes.
        """
        self._check_data_loaded()

        sns.pairplot(self.data)
        plt.suptitle("Pair Plot", y=1.02)
        plt.show()

    # ============================================================
    # 11. plot_histograms_numeric
    # ============================================================
    def plot_histograms_numeric(self):
        """
        Plots histograms of continuous numerical attributes.
        """
        self._check_data_loaded()

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        self.data[numeric_cols].hist(figsize=(12, 8), bins=20)
        plt.suptitle("Histograms of Numeric Attributes")
        plt.tight_layout()
        plt.show()

    # ============================================================
    # 12. plot_histograms_categorical
    # ============================================================
    def plot_histograms_categorical(self):
        """
        Plots histograms of nominal (categorical) attributes.
        """
        self._check_data_loaded()

        cat_cols = self.data.select_dtypes(include=["object", "category"]).columns

        for col in cat_cols:
            plt.figure(figsize=(6, 4))
            self.data[col].value_counts().plot(kind="bar")
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()

    # ============================================================
    # 13. plot_boxPlot
    # ============================================================
    def plot_boxPlot(self):
        """
        Plots box-plots of all numeric attributes.
        """
        self._check_data_loaded()

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        plt.figure(figsize=(12, 8))
        self.data[numeric_cols].boxplot()
        plt.title("Box Plot of Numeric Attributes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # ============================================================
    # 14. sample
    # ============================================================
    def sample(self, reduction_factor):
        """
        Randomly samples the dataset based on a reduction factor (0.0 to 1.0).

        Parameters
        ----------
        reduction_factor : float
            Fraction of the dataset to keep after sampling.
            Example: 0.5 keeps 50% of the samples.
        """
        self._check_data_loaded()

        if not (0.0 < reduction_factor <= 1.0):
            raise ValueError("reduction_factor must be between 0.0 and 1.0")

        original_size = len(self.data)
        new_size = int(original_size * reduction_factor)

        self.data = self.data.sample(n=new_size, random_state=None).reset_index(drop=True)

        print(f"Sampled dataset: kept {new_size} out of {original_size} rows "
              f"({reduction_factor * 100:.1f}%).")

    # ============================================================
    # Internal Helper
    # ============================================================
    def _check_data_loaded(self):
        if self.data is None:
            raise ValueError("No dataset loaded. Call read_dataset() first.")
