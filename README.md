
# Model Documentation: Predictive Maintenance Models for AI4I 2020 Dataset

## Overview
- **Purpose**: These models predict machine failures in industrial equipment to enable predictive maintenance, reducing downtime and costs,ultimately improving operational efficiency.. Model 1 serves as a baseline with basic feature engineering, while Model 2 incorporates advanced engineered features to better capture failure conditions (e.g., TWF, HDF, PWF, OSF rules). Both models address class imbalance in the dataset (3.4% failure rate) and aim for high recall on failures.
- **Model Type**: Random Forest Classifier (ensemble of decision trees for handling non-linear relationships and feature interactions).
- **Version**: 
  - Model 1: v1.0 (Baseline with SMOTE oversampling and basic features).
  - Model 2: v2.0 (Enhanced with rule-based risk features for improved failure detection).
- **Author(s)**: [Oluwapelumi I. OJO]
- **Last Updated**: September 21, 2025.

**Key Insights from Development**:
- Dataset shows low linear correlations between features and target (e.g., Tool wear: 0.105, Torque: 0.191), making Random Forest suitable for capturing non-linear patterns.
- Model 1 performs well on holdout test data but struggles on synthetic data, indicating potential overfitting or misalignment with real failure rules.
- Model 2 improves generalization by encoding domain-specific rules (e.g., OSF thresholds varying by product type), leading to better AUC and recall on both test and synthetic data.

---

## Dataset
- **Source**: UCI Machine Learning Repository - AI4I 2020 Predictive Maintenance Dataset (associated with Matzka, S. (2020) paper). Publicly available under CC BY 4.0 license.
- **Size**: 10,000 samples (rows), 14 original features (columns).
- **Features**: 
  - Categorical: Product ID (dropped), Type (L/M/H for quality variants; one-hot encoded as Type_H, Type_L, Type_M).
  - Numerical: Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min].
  - Binary Indicators: TWF (tool wear failure), HDF (heat dissipation failure), PWF (power failure), OSF (overstrain failure), RNF (random failure).
  - Target: Machine failure (binary: 0/1, imbalanced with ~3.4% positives).
- **Preprocessing**:
  - No missing values.
  - Exploratory Data Analysis (EDA): performed to identify distributions, correlations, and patterns
  - One-hot encoding for 'Type'.
  - Feature Engineering (Model 1): Temp_Diff = Process temperature [K] - Air temperature [K]; Power = Torque [Nm] * Rotational speed [rpm] * (2 * π / 60); Strain = Tool wear [min] * Torque [Nm].
  - Feature Engineering (Model 2): Temp_diff (same as above); Power [W] (same); Wear_Torque = Tool wear [min] * Torque [Nm]; TWF_risk = 1 if Tool wear [min] in [200, 240] else 0; OSF_risk = 1 if Wear_Torque exceeds type-specific thresholds (L: 11000, M: 12000, H: 13000) else 0.
  - Train-Test Split: 80% train (8000 samples), 20% test (2000 samples), stratified on target.
  - Handling Imbalance: SMOTE oversampling on training set + class_weight='balanced' in the model.
- **Challenges**: Severe class imbalance (339 failures out of 10,000). Failures depend on complex rules (e.g., HDF if Temp_diff < 8.6K and speed < 1380rpm; PWF if Power <3500W or >9000W; OSF based on Strain thresholds by Type). Synthetic test data (CavekXsync2025.csv, Mircavek25.csv) may not perfectly align with these rules, leading to performance drops.

**Dataset Statistics**:
| Statistic | Value |
|-----------|-------|
| Total Samples | 10,000 |
| Failures (1) | 339 (3.4%) |
| Non-Failures (0) | 9,661 (96.6%) |
| Features After Engineering (Model 1) | 11 (original + Temp_Diff, Power, Strain, one-hot Type) |
| Features After Engineering (Model 2) | 18 (original + Temp_diff, Power [W], Wear_Torque, TWF_risk, OSF_risk, one-hot Type) |

---

## Model Architecture
- **Environment**: Jupyter Notebook (Anaconda).
- **Framework**: Scikit-learn (Python 3.12).
- **Model**: RandomForestClassifier.
- **Hyperparameters** (Both Models):
  - n_estimators: 200 (number of trees).
  - max_depth: 10 (limits tree depth to prevent overfitting).
  - class_weight: 'balanced' (adjusts weights inversely proportional to class frequencies).
  - random_state: 12 (for reproducibility).
  - Other defaults: criterion='gini', min_samples_split=2, etc.
- **Parameters**: ~1-2 million (ensemble of 200 trees, each up to depth 10).
- **Diagram (Conceptual)**: Ensemble of decision trees, each voting on failure prediction. Features like Torque and Tool wear are key splitters due to their correlations.

**Feature Order (Input to Model)**:
- Model 1: ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type_H', 'Type_L', 'Type_M', 'Temp_Diff', 'Power', 'Strain'].
- Model 2: ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Temp_diff', 'Power [W]', 'Wear_Torque', 'TWF_risk', 'OSF_risk', 'Type_H', 'Type_L', 'Type_M'].

---

## Training
- **Loss Function**: Gini impurity (for tree splits; no explicit loss as it's a classifier).
- **Training Process**:
  1. Load dataset and engineer features.
  2. One-hot encode 'Type'.
  3. Split into X (features) and y (Machine failure).
  4. Apply SMOTE to oversample minority class in training set.
  5. Fit RandomForestClassifier.
- **Challenges**: Overfitting to training data in Model 1 (high test metrics but poor synthetic generalization). Addressed in Model 2 with rule-based features encoding domain knowledge (e.g., OSF_risk).

---

## Evaluation
- **Metrics**: Accuracy, ROC-AUC (primary for imbalance), Confusion Matrix, Precision/Recall/F1 (focus on class 1 recall).
- **Results on Holdout Test Set (20% of AI4I 2020)**:

  **Model 1**:
  - Accuracy: 0.98
  - ROC-AUC: 0.919
  - Confusion Matrix: [[1902, 30], [10, 58]]
  - Classification Report:
    | Class | Precision | Recall | F1-Score | Support |
    |-------|-----------|--------|----------|---------|
    | 0     | 0.99      | 0.98   | 0.99     | 1932    |
    | 1     | 0.66      | 0.85   | 0.74     | 68      |
    - Macro Avg: Precision 0.83, Recall 0.92, F1 0.87

  **Model 2**:
  - Accuracy: 0.996
  - ROC-AUC: 0.977
  - Confusion Matrix: [[1927, 5], [3, 65]]
  - Classification Report:
    | Class | Precision | Recall | F1-Score | Support |
    |-------|-----------|--------|----------|---------|
    | 0     | 1.00      | 1.00   | 1.00     | 1932    |
    | 1     | 0.93      | 0.96   | 0.94     | 68      |
    - Macro Avg: Precision 0.96, Recall 0.98, F1 0.97

- **Results on Synthetic Data** (CavekXsync2025.csv and Mircavek25.csv, 2000 samples each):

  **Model 1 on CavekXsync2025**:
  - Accuracy: 0.5625 | ROC-AUC: 0.522 | Confusion: [[1021, 79], [796, 104]] | Recall (Class 1): 0.12

  **Model 1 on Mircavek25**:
  - Accuracy: 0.8575 | ROC-AUC: 0.568 | Confusion: [[1698, 236], [49, 17]] | Recall (Class 1): 0.26

  **Model 2 on CavekXsync2025**:
  - Accuracy: 0.839 | ROC-AUC: 0.824 | Confusion: [[1069, 31], [291, 609]] | Recall (Class 1): 0.68

  **Model 2 on Mircavek25**:
  - Accuracy: 0.9105 | ROC-AUC: 0.815 | Confusion: [[1774, 160], [19, 47]] | Recall (Class 1): 0.71

- **Comparison**: Model 2 outperforms Model 1 on all metrics, especially on synthetic data (higher recall for failures: 0.68-0.71 vs. 0.12-0.26), due to better capture of failure rules.

---

## Deployment
- **Environment**: Python 3.12 with scikit-learn, pandas, numpy (via Anaconda).
- **Inference Time**: <1ms per prediction (ensemble of 200 shallow trees).
- **Dependencies**: scikit-learn==1.5, pandas==2.2, numpy==1.26, imbalanced-learn (for SMOTE), joblib (for saving).
- **Instructions**:
  1. Install dependencies: `pip install scikit-learn pandas numpy imbalanced-learn joblib`.
  2. Load model: `import joblib; model_data = joblib.load('model_file.pkl'); model = model_data['model']`.
  3. Prepare input DataFrame with features in the order from model_data['features_order'].
  4. Predict: `predictions = model.predict(new_data)`.
- **Challenges**: Ensure input data matches engineered features and types (use model_data['features_types'] for validation). Model may underperform if new data deviates from AI4I rules.

**Saved Files**:
- Model 1: ai4i2020_rfc_M1.pkl (includes model, feature_order, feature_types).
- Model 2: ai4i2020_rfc_M2.pkl (same).

---
## Access the Hosted App
Try the interactive demo live: [https://predictive-maintenance-ai4i2020.streamlit.app/](https://predictive-maintenance-ai4i2020.streamlit.app/)

Input machine stats and get instant "Healthy" vs. "Faulty" predictions!
---
## Reproducibility
- **Code Repository**:[link](https://github.com/0luwa-Pelumi/PredictiveMaintenance-AI4I2020.git)
- **Random Seed**: 12 (set in model and splits).
- **Environment Setup**:
  ```bash
  conda create -n pred_maint python=3.12
  conda activate pred_maint
  pip install scikit-learn pandas numpy imbalanced-learn joblib matplotlib seaborn
  ```
- **Run Instructions**: Download dataset from UCI, place at path in notebooks, execute cells sequentially in ai4i2020Model1.ipynb and 
---
## Access the Hosted App
Try the interactive demo live: [https://predictive-maintenance-ai4i2020.streamlit.app/](https://predictive-maintenance-ai4i2020.streamlit.app/)

Input machine stats and get instant "Healthy" vs. "Faulty" predictions!
---

## Future Improvements
- **Potential Enhancements**: Incorporate more rules (e.g., HDF_risk for Temp_diff <8.6K and low speed; PWF_risk for extreme Power). Experiment with XGBoost for better imbalance handling. Add time-series features if sequential data available. Validate on real industrial data.
- **Known Limitations**: Relies on synthetic rules; may not generalize to unseen failure modes. High false positives/negatives on mismatched synthetic data. No hyperparameter tuning (e.g., via GridSearchCV).

---

## References
- Dataset: Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. [Link](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset).
- Libraries: scikit-learn (Pedregosa et al., 2011), imbalanced-learn (Lemaître et al., 2017).
- Methodology: Inspired by domain rules from dataset description (e.g., failure thresholds).
