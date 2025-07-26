# Loan-Approval-Prediction
# Overview
This project builds a binary classification model to predict whether a loan application will be approved or rejected based on customer and financial attributes. Special focus is placed on handling imbalanced data, evaluating using precision, recall, and F1-score, and comparing different models including Logistic Regression, Decision Tree, and Random Forest.

# Dataset
- Source: Kaggle - Loan Approval Prediction Dataset
- File: loan_approval_dataset.csv
- Target Variable: loan_status (Approved / Rejected)
- Features Include:
  - income_annum, loan_amount, loan_term, cibil_score
  - self_employed, education, residential_assets_value, etc.

# Objectives
1. Clean and preprocess the dataset (handle missing values and encode categorical data).
2. Split the dataset for training and testing.
3. Handle class imbalance using SMOTE.
4. Train multiple classification models:
   - Random Forest
   - Logistic Regression
   - Decision Tree
5. Evaluate model performance using:
  - Precision, Recall, F1-score
  - Confusion Matrix visualization
6. Compare results across models.

# Tools & Libraries
- Python
- pandas
- numpy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn (for SMOTE)



Key Steps & Results
- Data Preprocessing
  - Missing values filled using forward fill
  - Categorical features encoded with LabelEncoder
  - Target variable loan_status: encoded to 0 (Rejected), 1 (Approved)
- Train-Test Split
  - 80% Training, 20% Testing
  - Stratified sampling to preserve class distribution
- Handling Imbalanced Data
  - Used SMOTE to balance the minority class in training data
    - Before: [2125 Rejected, 1290 Approved]
    - After: [2125 Rejected, 2125 Approved]
- Model Training and Evaluation
  - Random Forest
    - Accuracy: 98%
    - Precision/Recall (Approved): 0.98 / 0.96
  - Logistic Regression
    - Accuracy: 93%
    - Precision/Recall (Approved): 0.91 / 0.90
  - Decision Tree
    - Accuracy: 98%
    - Precision/Recall (Approved): 0.95 / 0.99
- Confusion Matrices
  - Visualized for all models using Seaborn heatmaps.

