# Credit Risk Fairness Analysis

![screenshot-localhost_8888-2025 04 02-13_10_20](https://github.com/user-attachments/assets/7024615a-1ba5-4960-8169-a5f9f38291b1)

### Project Overview

This project evaluated machine learning models for predicting credit risk while assessing their fairness across gender groups. This analysis highlights the trade-off between accuracy and fairness.

This project analyzes the German Credit Dataset to:

- Predict credit risk (good vs. bad) using machine learning models.
- Audit fairness across gender groups (male/female) using: Fairlearn (demographic parity, equalized odds) and Aequitas (disparity in false positives/negatives)

### Dataset

The dataset contains credit application information, including applicant demographics, financial status, and loan outcomes. Key variables include:

- Age
- Gender
- Credit History
- Employment Duration
- Loan Amount
- Default (Target Variable: 1 = Default, 0 = No Default)

### Methodology

1. Data Preprocessing
- Cleaned missing values in Saving accounts/Checking account.
- Encoded categorical features:Label Encoding for ordinal features (e.g., account types). One-Hot Encoding for nominal features (e.g., Sex, Housing).
- Balanced class imbalance using SMOTE.

2. Model Training

- Logistic Regression
- Random Forest
- XGBoost

3. Fairness Evaluation

Fairlearn Metrics:
- Demographic Parity Difference (<0.1 is fair)
- Equalized Odds Difference (<0.1 is fair)
- Disparate Impact Ratio (0.8–1.2 is fair)

2. Aequitas Audit:
- False Positive Rate (FPR) Disparity
- False Discovery Rate (FDR) Disparity

### Results

Demographic Parity Difference
- Logistic Regression: 6.7%
- XGBoost: 9.1%
- Not Fair (Goal < 5%)

Equalized Odds Difference
- Logistic Regression: 12.0%
- XGBoost: 11.0%
- Not Fair (Goal < 10%)

Disparate Impact Ratio
- Logistic Regression: 1.11
- XGBoost: 1.15
- Fair (Range: 0.8–1.2)

False Positive Rate (FPR) Disparity for Women
- Logistic Regression: 1.29×
- XGBoost: 1.0×
- Only XGBoost is fair

### Key Findings:

- Men were 6–9% more likely to be approved than women.
- Women faced 11–12.5% higher error rates (e.g., false rejections).
- XGBoost was most accurate but slightly less fair.

### Conclusion

While models met some fairness thresholds (e.g., disparate impact ratio), the disparities in error rates (higher false rejections for women) could have real-world consequences. Regular audits and transparency in model decisions are critical to ensure equitable outcomes.

### Source

![German Credit Data from Kaggle](https://www.kaggle.com/datasets/varunchawla30/german-credit-data)
