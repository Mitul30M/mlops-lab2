# Telecom Tower Failure Prediction

A machine learning pipeline to predict telecom tower failures within 48 hours using sensor data. Built with **RandomForestClassifier** and includes full data exploration, preprocessing, feature engineering, hyperparameter tuning, and model evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Command Line](#command-line)
- [Pipeline Steps](#pipeline-steps)
- [Feature Engineering](#feature-engineering)
- [Model Performance](#model-performance)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

This project builds a predictive maintenance model for telecom towers. By analyzing real-time sensor data (temperature, battery voltage, power consumption, signal strength, fan speed, humidity, traffic load, and tower age), the model predicts whether a tower will fail within the next 48 hours.

### Key Features

- **End-to-end ML pipeline** from raw data to production-ready model
- **Comprehensive EDA** with visualizations
- **Automated feature engineering** (thermal stress, power efficiency, categorical encodings)
- **Hyperparameter tuning** via GridSearchCV with cross-validation
- **Full model evaluation** with confusion matrix, ROC curve, and classification report
- **Reproducible training script** with CLI arguments
- **Serialized model** for easy deployment

---

## Dataset

| Feature | Description | Range |
|---------|-------------|-------|
| `Tower_ID` | Unique tower identifier | 1–10,000 |
| `Temperature_C` | Ambient temperature (°C) | 20–70 |
| `Battery_Voltage` | Battery voltage (V) | 42–54 |
| `Power_Consumption_W` | Power draw (W) | 500–3000 |
| `Signal_Strength_Percent` | Signal quality (%) | 40–100 |
| `Fan_Speed_RPM` | Cooling fan speed (RPM) | 800–3500 |
| `Humidity_Percent` | Relative humidity (%) | 20–95 |
| `Traffic_Load` | Network traffic load | 100–5000 |
| `Tower_Age_Years` | Tower age (years) | 1–15 |
| `Failure_Within_48Hrs` | **Target**: 1 = failure, 0 = no failure | 0 or 1 |

**Target Distribution:** ~67% failures, ~33% no failures

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone or download the project
cd telecom-tower-failure-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=1.3.0 | Data manipulation |
| numpy | >=1.21.0 | Numerical computing |
| scikit-learn | >=1.0.0 | ML algorithms & evaluation |
| matplotlib | >=3.4.0 | Plotting |
| seaborn | >=0.11.0 | Statistical visualizations |

---

## Usage

### Quick Start

Run the training pipeline with default settings:

```bash
python train.py
```

This will:
1. Load `Telecom_Tower_Failure_Dataset_10000-1.csv`
2. Perform EDA and data preprocessing
3. Engineer new features
4. Train a RandomForestClassifier with GridSearchCV
5. Evaluate on test set
6. Save `model.pkl`, `metrics.json`, and `config.json` to the `output/` directory

### Jupyter Notebook

For interactive exploration and step-by-step execution:

```bash
jupyter notebook telecom_tower_failure_prediction.ipynb
```

The notebook includes:
- Data loading and exploration
- EDA visualizations (6 subplots)
- Data quality checks and outlier detection
- Feature engineering with explanations
- Model training with cross-validation
- Hyperparameter tuning
- Evaluation plots (confusion matrix, ROC curve)
- Feature importance analysis

### Command Line

The `train.py` script supports several CLI arguments:

```bash
python train.py --data path/to/dataset.csv --output my_output --test-size 0.25 --random-state 123
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `Telecom_Tower_Failure_Dataset_10000-1.csv` | Path to input CSV |
| `--output` | `output` | Output directory for artifacts |
| `--test-size` | `0.2` | Proportion of data for testing (0.0–1.0) |
| `--random-state` | `42` | Random seed for reproducibility |

---

## Pipeline Steps

```
Raw Data
    │
    ▼
┌─────────────────┐
│  Data Loading   │  → Load CSV, verify structure
└─────────────────┘
    │
    ▼
┌─────────────────┐
│      EDA        │  → Statistical summary, distributions, correlations
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Data Cleaning   │  → Check missing values, duplicates, outliers, ranges
└─────────────────┘
    │
    ▼
┌─────────────────┐
│Feature Engineer │  → Create derived features, encode categories
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Train-Test     │  → Stratified split (80/20)
│     Split       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Model Training  │  → RandomForest + GridSearchCV (3-fold)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   Evaluation    │  → Accuracy, Precision, Recall, F1, ROC-AUC
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Save Artifacts │  → model.pkl, metrics.json, config.json
└─────────────────┘
```

---

## Feature Engineering

Four new features are engineered from the raw sensor data:

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `Temp_Humidity_Interaction` | `Temperature_C × Humidity_Percent / 100` | Thermal stress indicator — high temp + high humidity increases failure risk |
| `Power_Efficiency` | `Power_Consumption_W / (Signal_Strength_Percent + 1)` | Watts per unit signal — inefficiency may indicate hardware degradation |
| `Age_Category_Encoded` | `cut(Tower_Age_Years, [0,5,10,15])` | Categorical age: New (1–5), Mid (6–10), Old (11–15) |
| `Temp_Category_Encoded` | `cut(Temperature_C, [0,35,55,100])` | Categorical temp: Cool (<35°C), Moderate (35–55°C), Hot (>55°C) |

**Final feature set:** 12 features (8 original + 4 engineered)

---

## Model Performance

### Best Hyperparameters (GridSearchCV)

```json
{
  "max_depth": 10,
  "max_features": "sqrt",
  "min_samples_leaf": 1,
  "min_samples_split": 5,
  "n_estimators": 200
}
```

### Test Set Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 1.0000 |
| **Precision** | 1.0000 |
| **Recall** | 1.0000 |
| **F1-Score** | 1.0000 |
| **ROC-AUC** | 1.0000 |

> **Note:** Perfect scores (1.0000) indicate the dataset has very strong predictive signals or may be synthetic. In a production scenario with real-world noise, expect lower scores. Consider adding regularization, collecting more diverse data, or using ensemble methods for robustness.

### Feature Importance (Top 5)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `Battery_Voltage` | 24.9% |
| 2 | `Signal_Strength_Percent` | 24.4% |
| 3 | `Temperature_C` | 20.1% |
| 4 | `Fan_Speed_RPM` | 15.7% |
| 5 | `Temp_Category_Encoded` | 5.9% |

---

## Output Files

After running `train.py`, the following files are generated in the output directory:

| File | Description |
|------|-------------|
| `model.pkl` | Serialized trained RandomForest model (pickle) |
| `metrics.json` | Complete performance metrics, confusion matrix, feature importance |
| `config.json` | Feature names and model metadata for inference |

### Loading the Model for Inference

```python
import pickle
import pandas as pd

# Load model
with open('output/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare input (must match training features)
input_data = pd.DataFrame({
    'Temperature_C': [45.0],
    'Battery_Voltage': [48.5],
    'Power_Consumption_W': [1500],
    'Signal_Strength_Percent': [75],
    'Fan_Speed_RPM': [2500],
    'Humidity_Percent': [60],
    'Traffic_Load': [2000],
    'Tower_Age_Years': [8],
    'Temp_Humidity_Interaction': [27.0],      # 45 * 60 / 100
    'Power_Efficiency': [19.74],               # 1500 / (75 + 1)
    'Age_Category_Encoded': [1],               # Mid age
    'Temp_Category_Encoded': [1]               # Moderate temp
})

# Predict
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)[:, 1]

print(f"Failure predicted: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Failure probability: {probability[0]:.4f}")
```

---

## Project Structure

```
telecom-tower-failure-prediction/
│
├── telecom_tower_failure_prediction.ipynb   # Interactive Jupyter notebook
├── train.py                                 # Standalone training script
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
│
├── data/
│   └── Telecom_Tower_Failure_Dataset_10000-1.csv   # Input dataset
│
└── output/                                   # Generated artifacts
    ├── model.pkl                             # Trained model
    ├── metrics.json                          # Performance metrics
    ├── config.json                           # Model configuration
    ├── eda_visualizations.png                # EDA plots
    ├── feature_importance.png                # Feature importance chart
    └── model_evaluation.png                  # Confusion matrix & ROC curve
```

---

## License

This project is provided as-is for educational and demonstration purposes.

---

## Contact

For questions or issues, please refer to the notebook (`telecom_tower_failure_prediction.ipynb`) for detailed, step-by-step explanations of each pipeline stage.
