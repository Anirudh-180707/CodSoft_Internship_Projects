# Credit Card Fraud Detection

## Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques. Due to the highly imbalanced nature of fraud datasets, appropriate preprocessing and class imbalance handling methods were applied before training classification models.

The objective is to accurately classify transactions as either fraudulent or genuine while evaluating model performance using standard classification metrics.

## Features

* Data preprocessing and cleaning
* Feature scaling and normalization
* Handling class imbalance
* Train-test data splitting
* Logistic Regression model
* Random Forest model
* Performance evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix
  * Classification Report

## Dataset

The original dataset contains anonymized credit card transactions made by European cardholders.

**Original Dataset Source (Kaggle):**
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Note About Dataset File

The original dataset file is relatively large and may not be viewable directly on GitHub due to file size limitations.

For repository upload purposes, a reduced version of the dataset has been included. Users who wish to work with the complete dataset can download it from the Kaggle link provided above.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)

## Project Structure

```text
Credit_Card_Fraud_Detection
│
├── credit_card_fraud_detection.py
├── creditcard_under_25mb.csv
└── README.md
```

## How to Run

1. Clone the repository.

```bash
git clone <repository-url>
```

2. Navigate to the project folder.

```bash
cd Credit_Card_Fraud_Detection
```

3. Install required libraries.

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

4. Run the project.

```bash
python credit_card_fraud_detection.py
```

## Expected Output

The program displays:

* Dataset information
* Class distribution
* Model evaluation metrics
* Confusion matrices
* Classification reports for both models

## Conclusion

This project demonstrates how Machine Learning can be applied to detect fraudulent financial transactions. By preprocessing data, addressing class imbalance, and evaluating multiple classification models, the system can effectively distinguish between legitimate and fraudulent transactions.
