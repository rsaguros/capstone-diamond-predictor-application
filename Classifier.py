# Classifier.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix


class Classifier:
    def __init__(self):
        """
        Classifier class supporting multiple estimators:
        - Logistic Regression
        - KNN
        - Decision Tree
        - Random Forest
        - SVC
        - ANN (MLPClassifier)
        """
        self.model = None
        self.is_trained = False

    # ============================================================
    # 1. Logistic Regression
    # ============================================================
    def logistic_regression(self):
        """
        Creates a logistic regression classifier.
        """
        self.model = LogisticRegression(max_iter=500)
        print("Initialized Logistic Regression model.")

    # ============================================================
    # 2. KNN Classifier
    # ============================================================
    def knn_classifier(self, X_train, y_train, X_test, y_test, k_values=[1, 3, 5, 7, 9]):
        """
        Trains KNN with different K values, prints the optimal K,
        and returns accuracy for predicting the 'cut' label.
        """
        best_k = None
        best_score = -1

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            score = knn.score(X_test, y_test)

            print(f"K={k}, Accuracy={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k

        print(f"Optimal K: {best_k}")

        # Set the best model as the active model
        self.model = KNeighborsClassifier(n_neighbors=best_k)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        return best_score

    # ============================================================
    # 3. Decision Tree
    # ============================================================
    def decision_tree(
            self, X_train, y_train, X_test, y_test,
            criteria=["gini", "entropy", "log_loss"]
        ):
        """
        Trains a decision tree with different criteria and returns accuracy.
        """
        best_criterion = None
        best_score = -1

        for c in criteria:
            dt = DecisionTreeClassifier(criterion=c)
            dt.fit(X_train, y_train)
            score = dt.score(X_test, y_test)

            print(f"Criterion={c}, Accuracy={score:.4f}")

            if score > best_score:
                best_score = score
                best_criterion = c

        print(f"Optimal Criterion: {best_criterion}")

        self.model = DecisionTreeClassifier(criterion=best_criterion)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        return best_score

    # ============================================================
    # 4. Random Forest
    # ============================================================
    def random_forest(
            self, X_train, y_train, X_test, y_test,
            criteria=["gini", "entropy"],
            n_estimators_list=[50, 100, 150],
        ):
        """
        Trains a random forest with different criteria and estimators.
        """
        best_params = None
        best_score = -1

        for c in criteria:
            for n in n_estimators_list:
                rf = RandomForestClassifier(criterion=c, n_estimators=n, random_state=42, class_weight="balanced")
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)

                print(f"Criterion={c}, Estimators={n}, Accuracy={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_params = (c, n)

        print(f"Optimal Random Forest: Criterion={best_params[0]}, Estimators={best_params[1]}")

        self.model = RandomForestClassifier(
            criterion=best_params[0],
            n_estimators=best_params[1],
            random_state=42, 
            class_weight="balanced",
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        return best_score

    # ============================================================
    # 5. SVC
    # ============================================================
    def svc(self, X_train, y_train, X_test, y_test):
        """
        Trains an SVC and returns accuracy.
        """
        self.model = SVC()
        self.model.fit(X_train, y_train)
        self.is_trained = True

        score = self.model.score(X_test, y_test)
        print(f"SVC Accuracy: {score:.4f}")

        return score

    # ============================================================
    # 6. Artificial Neural Network (ANN)
    # ============================================================
    def ann(self, X_train, y_train, X_test, y_test,
            architectures=[(50,), (100,), (50, 50)],
            learning_rates=[0.001, 0.01],
            activations=["relu", "tanh"]):
        """
        Trains ANN with different architectures, learning rates, and activations.
        Returns the best accuracy.
        """
        best_score = -1
        best_params = None

        for arch in architectures:
            for lr in learning_rates:
                for act in activations:
                    ann = MLPClassifier(
                        hidden_layer_sizes=arch,
                        learning_rate_init=lr,
                        activation=act,
                        max_iter=500,
                    )
                    ann.fit(X_train, y_train)
                    score = ann.score(X_test, y_test)

                    print(f"Arch={arch}, LR={lr}, Act={act}, Accuracy={score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_params = (arch, lr, act)

        print(f"Best ANN: Arch={best_params[0]}, LR={best_params[1]}, Act={best_params[2]}")

        # Set the best model
        self.model = MLPClassifier(
            hidden_layer_sizes=best_params[0],
            learning_rate_init=best_params[1],
            activation=best_params[2],
            max_iter=500,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        return best_score

    # ============================================================
    # Unified API: fit, predict, score
    # ============================================================
    def fit(self, X_train, y_train):
        self._check_model_initialized()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Model trained.")

    def predict(self, X):
        self._check_trained()
        return self.model.predict(X)

    def score(self, X, y):
        self._check_trained()
        return accuracy_score(y, self.model.predict(X))

    # ============================================================
    # Confusion Matrix
    # ============================================================
    def plot_confusionMatrix(self, y_true, y_pred, class_names="auto"):
        """
        Plots and returns the confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

        return cm

    # ============================================================
    # Internal Helpers
    # ============================================================
    def _check_model_initialized(self):
        if self.model is None:
            raise ValueError("No estimator initialized. Call an estimator function first.")

    def _check_trained(self):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() or a training method first.")
