# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import argparse

# Pearson Correlation-based feature selection
def cor_selector(X, y, num_feats):
    """
    Select features based on Pearson correlation coefficients.

    Parameters:
    - X: DataFrame, feature matrix.
    - y: Series, target variable.
    - num_feats: int, number of top features to select.

    Returns:
    - cor_support: list, boolean mask of selected features.
    - cor_feature: list, names of selected features.
    """
    X_norm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)  # Normalize features
    cor_list = [np.corrcoef(X[col], y)[0, 1] for col in X_norm.columns]  # Calculate correlations
    feature_value = pd.DataFrame({'Feature': X_norm.columns, 'Correlation': np.abs(cor_list)}).sort_values(
        'Correlation', ascending=False
    )
    top_features = feature_value.iloc[:num_feats, :]  # Select top features
    cor_support = [col in top_features['Feature'].tolist() for col in X.columns]
    cor_feature = X.columns[cor_support].tolist()
    return cor_support, cor_feature

# Chi-squared test-based feature selection
def chi_squared_selector(X, y, num_feats):
    """
    Select features based on the chi-squared statistical test.

    Parameters:
    - X: DataFrame, feature matrix.
    - y: Series, target variable.
    - num_feats: int, number of top features to select.

    Returns:
    - chi_support: list, boolean mask of selected features.
    - chi_feature: list, names of selected features.
    """
    X_norm = MinMaxScaler().fit_transform(X)  # Normalize features
    chi_selector = SelectKBest(chi2, k=num_feats).fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()
    return chi_support, chi_feature

# Recursive Feature Elimination (RFE)
def rfe_selector(X, y, num_feats):
    """
    Select features using RFE with Logistic Regression as the base model.

    Parameters:
    - X: DataFrame, feature matrix.
    - y: Series, target variable.
    - num_feats: int, number of top features to select.

    Returns:
    - rfe_support: list, boolean mask of selected features.
    - rfe_feature: list, names of selected features.
    """
    X_norm = MinMaxScaler().fit_transform(X)  # Normalize features
    rfe_selector = RFE(estimator=LogisticRegression(random_state=42), n_features_to_select=num_feats, step=1).fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:, rfe_support].columns.tolist()
    return rfe_support, rfe_feature

# Embedded Logistic Regression (L1 regularization)
def embedded_log_reg_selector(X, y, num_feats):
    """
    Select features using Logistic Regression with L1 regularization.

    Parameters:
    - X: DataFrame, feature matrix.
    - y: Series, target variable.
    - num_feats: int, number of top features to select.

    Returns:
    - embedded_lr_support: list, boolean mask of selected features.
    - embedded_lr_feature: list, names of selected features.
    """
    X_norm = MinMaxScaler().fit_transform(X)  # Normalize features
    embedded_lr_selector = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
        max_features=num_feats
    ).fit(X_norm, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
    return embedded_lr_support, embedded_lr_feature

# Embedded Random Forest feature importance
def embedded_rf_selector(X, y, num_feats):
    """
    Select features using Random Forest feature importance.

    Parameters:
    - X: DataFrame, feature matrix.
    - y: Series, target variable.
    - num_feats: int, number of top features to select.

    Returns:
    - embedded_rf_support: list, boolean mask of selected features.
    - embedded_rf_feature: list, names of selected features.
    """
    X_norm = MinMaxScaler().fit_transform(X)  # Normalize features
    embedded_rf_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        max_features=num_feats
    ).fit(X_norm, y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    return embedded_rf_support, embedded_rf_feature

# Embedded LightGBM feature importance
def embedded_lgbm_selector(X, y, num_feats):
    """
    Select features using LightGBM feature importance.

    Parameters:
    - X: DataFrame, feature matrix.
    - y: Series, target variable.
    - num_feats: int, number of top features to select.

    Returns:
    - embedded_lgbm_support: list, boolean mask of selected features.
    - embedded_lgbm_feature: list, names of selected features.
    """
    X_norm = MinMaxScaler().fit_transform(X)  # Normalize features
    lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, random_state=42, verbosity=-1)
    embedded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats).fit(X_norm, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    return embedded_lgbm_support, embedded_lgbm_feature

# Data preprocessing
def preprocess_dataset(dataset):
    """
    Preprocess the dataset: handle missing values and encode categorical variables.

    Parameters:
    - dataset: DataFrame, the input dataset.

    Returns:
    - X: DataFrame, feature matrix.
    - y: Series, target variable.
    """
    dataset = dataset.dropna(axis=1)  # Drop columns with missing values
    y = dataset.iloc[:, -1]  # Assume the last column is the target variable
    X = dataset.iloc[:, :-1]  # Feature matrix
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)  # One-hot encoding
    return X, y

# Auto feature selection
def autoFeatureSelector(dataset_path, methods=[], num_output_features=10):
    """
    Perform automatic feature selection using multiple methods.

    Parameters:
    - dataset_path: DataFrame, input dataset.
    - methods: list, feature selection methods to apply (e.g., ['pearson', 'chi-square', ...]).
    - num_output_features: int, number of top features to select.

    Returns:
    - best_features: list, selected feature names.
    """
    X, y = preprocess_dataset(dataset_path)  # Preprocess dataset
    feature_name = list(X.columns)
    support_dict = {}
    feature_dict = {}

    # Apply specified feature selection methods
    for method in methods:
        if method == 'pearson':
            support, features = cor_selector(X, y, num_output_features)
        elif method == 'chi-square':
            support, features = chi_squared_selector(X, y, num_output_features)
        elif method == 'rfe':
            support, features = rfe_selector(X, y, num_output_features)
        elif method == 'log-reg':
            support, features = embedded_log_reg_selector(X, y, num_output_features)
        elif method == 'rf':
            support, features = embedded_rf_selector(X, y, num_output_features)
        elif method == 'lgbm':
            support, features = embedded_lgbm_selector(X, y, num_output_features)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        support_dict[method] = support
        feature_dict[method] = features

    # Create a dataframe summarizing feature selection results
    feature_selection_df = pd.DataFrame({'Feature': feature_name})
    for method, support in support_dict.items():
        feature_selection_df[method] = support
    feature_selection_df['Total'] = feature_selection_df.iloc[:, 1:].sum(axis=1)  # Count votes
    feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)

    # Select features with maximum votes
    best_features = feature_selection_df.head(num_output_features)['Feature'].tolist()
    print(feature_selection_df)  # Print summary table
    return best_features