# Diamonds ML Project

A modular machine‑learning pipeline built on the Diamonds dataset, demonstrating **classification**, **regression**, and **clustering** with a focus on clean architecture and interpretability.

## Features

- End‑to‑end ML workflows (preprocessing → modeling → evaluation → visualization)
- Modular design with dedicated classes for data handling, classification, regression, and clustering
- Hyperparameter sweeps for multiple algorithms
- PCA‑based cluster visualization with human‑readable cluster names
- Single‑diamond price prediction demo

## Project Structure

```
main.py          # Runs all ML scenarios
Analyzer.py      # Data loading, cleaning, encoding, EDA tools
Classifier.py    # Classification models + confusion matrix
Regressor.py     # Regression models + scoring metrics
Clustering.py    # K-Means, Agglomerative, Mean-Shift + elbow method
diamonds.csv     # Dataset can be downloaded from Kaggle or link below.
```
[Diamonds Dataset](https://www.kaggle.com/shivam2503/diamonds/download)

## Scenarios

### Classification  
Predicts **diamond cut** using:
- KNN (with automatic K selection)
- Logistic Regression, Decision Tree, Random Forest, SVC, ANN (optional)

Outputs:
- Accuracy
- Confusion matrix

### Regression  
Predicts **diamond price** using:
- Random Forest (default demo)
- Linear Regression, KNN, Decision Tree, SVR, ANN (optional)

Outputs:
- R², MSE, MAE, RMSE  
- Example single‑diamond prediction

### Clustering  
Unsupervised grouping using:
- K‑Means (elbow method)
- Agglomerative Clustering
- Mean‑Shift (sampled for speed)

Outputs:
- PCA scatterplots  
- Interpreted PCA axis labels  
- Human‑readable cluster names (e.g., “Large Expensive (Ideal, VVS1)”)

## Installation

```
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Run the Project

```
python main.py
```

Each scenario pauses before continuing.

## Extending

- Swap in different models (Classifier/Regressor)
- Add new clustering algorithms
- Add or integrate a front-end UI.
