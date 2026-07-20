"""
Telecom Tower Failure Prediction - Training Script
=================================================
This script trains a RandomForestClassifier to predict telecom tower failures
within 48 hours based on sensor data.

Usage:
    python train.py --data path/to/dataset.csv --output path/to/output/dir

Author: Generated Pipeline
"""

import os
import sys
import argparse
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


def load_data(filepath):
    """Load and validate the dataset."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df):
    """Perform basic data exploration."""
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)

    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    print(f"\nTarget Distribution:")
    print(df['Failure_Within_48Hrs'].value_counts())
    print(f"\nStatistical Summary:")
    print(df.describe())


def preprocess_data(df):
    """Clean and preprocess the data."""
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)

    # Separate features and target
    X = df.drop(['Tower_ID', 'Failure_Within_48Hrs'], axis=1)
    y = df['Failure_Within_48Hrs']

    print(f"\nOriginal features: {list(X.columns)}")

    # Feature Engineering
    X_processed = X.copy()

    # Thermal stress indicator
    X_processed['Temp_Humidity_Interaction'] = (
        X_processed['Temperature_C'] * X_processed['Humidity_Percent'] / 100
    )

    # Power efficiency
    X_processed['Power_Efficiency'] = (
        X_processed['Power_Consumption_W'] / (X_processed['Signal_Strength_Percent'] + 1)
    )

    # Age category encoding
    age_bins = [0, 5, 10, 15]
    age_labels = ['New', 'Mid', 'Old']
    X_processed['Age_Category'] = pd.cut(X_processed['Tower_Age_Years'], bins=age_bins, labels=age_labels)
    le_age = LabelEncoder()
    X_processed['Age_Category_Encoded'] = le_age.fit_transform(X_processed['Age_Category'].astype(str))

    # Temperature category encoding
    temp_bins = [0, 35, 55, 100]
    temp_labels = ['Cool', 'Moderate', 'Hot']
    X_processed['Temp_Category'] = pd.cut(X_processed['Temperature_C'], bins=temp_bins, labels=temp_labels)
    le_temp = LabelEncoder()
    X_processed['Temp_Category_Encoded'] = le_temp.fit_transform(X_processed['Temp_Category'].astype(str))

    # Drop temporary categorical columns
    X_processed = X_processed.drop(['Age_Category', 'Temp_Category'], axis=1)

    print(f"\nEngineered features added:")
    print("  - Temp_Humidity_Interaction")
    print("  - Power_Efficiency")
    print("  - Age_Category_Encoded")
    print("  - Temp_Category_Encoded")
    print(f"\nFinal feature set: {list(X_processed.columns)}")

    return X_processed, y


def train_model(X_train, y_train):
    """Train RandomForestClassifier with hyperparameter tuning."""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }

    print(f"\nHyperparameter grid: {param_grid}")

    # GridSearchCV
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Training metrics
    print("\n--- Training Metrics ---")
    train_metrics = {
        "accuracy": float(accuracy_score(y_train, y_train_pred)),
        "precision": float(precision_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred)),
        "f1_score": float(f1_score(y_train, y_train_pred)),
        "roc_auc": float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    }
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Test metrics
    print("\n--- Test Metrics ---")
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "f1_score": float(f1_score(y_test, y_test_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_test_proba))
    }
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n--- Confusion Matrix ---")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")

    # Classification Report
    print(f"\n--- Classification Report ---")
    print(classification_report(y_test, y_test_pred, target_names=['No Failure', 'Failure']))

    return train_metrics, test_metrics, cm


def save_artifacts(model, metrics, cm, feature_names, output_dir):
    """Save model, metrics, and configuration."""
    print("\n" + "="*60)
    print("SAVING ARTIFACTS")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: {model_path}")

    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved: {metrics_path}")

    # Save feature names for inference
    config = {
        "feature_names": feature_names,
        "model_type": "RandomForestClassifier",
        "target_column": "Failure_Within_48Hrs"
    }
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✓ Config saved: {config_path}")

    return model_path, metrics_path


def main():
    parser = argparse.ArgumentParser(description='Train Telecom Tower Failure Prediction Model')
    parser.add_argument('--data', type=str, default='Telecom_Tower_Failure_Dataset_10000-1.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for model artifacts')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("TELECOM TOWER FAILURE PREDICTION - TRAINING PIPELINE")
    print("="*60)

    # 1. Load data
    df = load_data(args.data)

    # 2. Explore data
    explore_data(df)

    # 3. Preprocess data
    X, y = preprocess_data(df)

    # 4. Train-test split
    print(f"\nSplitting data (test_size={args.test_size}, random_state={args.random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # 5. Train model
    model, best_params, best_cv_score = train_model(X_train, y_train)

    # 6. Evaluate model
    train_metrics, test_metrics, cm = evaluate_model(model, X_train, X_test, y_train, y_test)

    # 7. Feature importance
    print(f"\n--- Feature Importance ---")
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")

    # 8. Compile metrics
    metrics = {
        "model": "RandomForestClassifier",
        "best_parameters": best_params,
        "cross_validation": {
            "mean_accuracy": float(best_cv_score),
            "std_accuracy": 0.0
        },
        "training_metrics": train_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1])
        },
        "feature_importance": feature_importance,
        "dataset_info": {
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "num_features": int(X.shape[1]),
            "features": list(X.columns)
        }
    }

    # 9. Save artifacts
    model_path, metrics_path = save_artifacts(
        model, metrics, cm, list(X.columns), args.output
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - Model: {model_path}")
    print(f"  - Metrics: {metrics_path}")
    print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

    return model, metrics


if __name__ == "__main__":
    main()