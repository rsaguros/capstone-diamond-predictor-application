# Regressor.py

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


class Regressor:
    def __init__(self):
        self.model = None
        self.is_trained = False

    # ============================================================
    # 1. Linear Regression
    # ============================================================
    def linear_regression(self):
        self.model = LinearRegression()
        print("Initialized Linear Regression model.")

    # ============================================================
    # 2. KNN Regressor
    # ============================================================
    def knn_regressor(self, X_train, y_train, X_test, y_test, k_values=[1, 3, 5, 7, 9]):
        """
        Train KNN with different K values.
        Return list of scores for predicting 'price'.
        """
        scores = []
        best_k = None
        best_score = -np.inf

        for k in k_values:
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X_train, y_train)
            score = knn.score(X_test, y_test)
            scores.append(score)

            print(f"K={k}, Score={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k

        print(f"Optimal K: {best_k}")

        # Set best model
        self.model = KNeighborsRegressor(n_neighbors=best_k)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        return scores

    # ============================================================
    # 3. Decision Tree Regressor
    # ============================================================
    def decision_tree(self, X_train, y_train, X_test, y_test,
                      criteria=["squared_error", "friedman_mse", "absolute_error"]):

        scores = []
        best_criterion = None
        best_score = -np.inf

        for c in criteria:
            dt = DecisionTreeRegressor(criterion=c)
            dt.fit(X_train, y_train)
            score = dt.score(X_test, y_test)
            scores.append(score)

            print(f"Criterion={c}, Score={score:.4f}")

            if score > best_score:
                best_score = score
                best_criterion = c

        print(f"Optimal Criterion: {best_criterion}")

        self.model = DecisionTreeRegressor(criterion=best_criterion)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        return scores

    # ============================================================
    # 4. Random Forest Regressor
    # ============================================================
    def random_forest(self, X_train, y_train, X_test, y_test,
                      criteria=["squared_error", "absolute_error"],
                      n_estimators_list=[5, 10, 20]):

        scores = []
        best_params = None
        best_score = -np.inf

        for c in criteria:
            for n in n_estimators_list:
                rf = RandomForestRegressor(criterion=c, n_estimators=n)
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                scores.append(score)

                print(f"Criterion={c}, Estimators={n}, Score={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_params = (c, n)

        print(f"Optimal Random Forest: Criterion={best_params[0]}, Estimators={best_params[1]}")

        self.model = RandomForestRegressor(
            criterion=best_params[0],
            n_estimators=best_params[1],
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        return scores

    # ============================================================
    # 5. SVR
    # ============================================================
    def svr(self, X_train, y_train, X_test, y_test, kernel="rbf"):
        svr = SVR(kernel=kernel)
        svr.fit(X_train, y_train)
        score = svr.score(X_test, y_test)

        print(f"SVR Score: {score:.4f}")

        self.model = svr
        self.is_trained = True

        return [score]

    # ============================================================
    # 6. ANN Regressor
    # ============================================================
    def ann(self, X_train, y_train, X_test, y_test,
            architectures=[(50,), (100,), (50, 50)],
            learning_rates=[0.001, 0.01],
            activations=["relu", "tanh"]):

        scores = []
        best_params = None
        best_score = -np.inf

        for arch in architectures:
            for lr in learning_rates:
                for act in activations:
                    ann = MLPRegressor(
                        hidden_layer_sizes=arch,
                        learning_rate_init=lr,
                        activation=act,
                        max_iter=500
                    )
                    ann.fit(X_train, y_train)
                    score = ann.score(X_test, y_test)
                    scores.append(score)

                    print(f"Arch={arch}, LR={lr}, Act={act}, Score={score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_params = (arch, lr, act)

        print(f"Best ANN: Arch={best_params[0]}, LR={best_params[1]}, Act={best_params[2]}")

        self.model = MLPRegressor(
            hidden_layer_sizes=best_params[0],
            learning_rate_init=best_params[1],
            activation=best_params[2],
            max_iter=500
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        return scores

    # ============================================================
    # Unified API
    # ============================================================
    def fit(self, X_train, y_train):
        self._check_model_initialized()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Model trained.")

    def predict(self, X):
        self._check_trained()
        return self.model.predict(X)

    def score(self, X, y_true, metric):
        """
        metric ∈ {'r2', 'mse', 'mae', 'rmse'}
        """
        self._check_trained()
        y_pred = self.model.predict(X)

        if metric == "r2":
            return r2_score(y_true, y_pred)
        elif metric == "mse":
            return mean_squared_error(y_true, y_pred)
        elif metric == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif metric == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        else:
            raise ValueError("Invalid metric. Choose from: 'r2', 'mse', 'mae', 'rmse'.")

    # ============================================================
    # Internal Helpers
    # ============================================================
    def _check_model_initialized(self):
        if self.model is None:
            raise ValueError("No estimator initialized. Call an estimator function first.")

    def _check_trained(self):
        if not self.is_trained:
            raise ValueError("Model is not trained. Call fit() or a training method first.")