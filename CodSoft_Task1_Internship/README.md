# Titanic Survival Prediction Using Machine Learning

## Project Overview

The Titanic Survival Prediction project aims to build a Machine Learning model that predicts whether a passenger survived the Titanic disaster based on passenger information such as age, gender, passenger class, fare, and family details.

This project demonstrates the complete machine learning workflow, including data preprocessing, model training, prediction, and performance evaluation.

---

## Dataset Information

The dataset contains passenger details such as:

* Passenger Class (Pclass)
* Gender (Sex)
* Age
* Siblings/Spouses Aboard (SibSp)
* Parents/Children Aboard (Parch)
* Fare
* Embarked Port
* Survival Status

### Target Variable

* 0 = Did Not Survive
* 1 = Survived

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn

---

## Machine Learning Algorithm

### Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## Project Workflow

### 1. Data Loading

Loaded the Titanic dataset using Pandas.

### 2. Data Preprocessing

* Removed unnecessary columns
* Handled missing values
* Encoded categorical variables

### 3. Feature Selection

Selected relevant passenger information for prediction.

### 4. Train-Test Split

Split the dataset into:

* 80% Training Data
* 20% Testing Data

### 5. Model Training

Trained a Random Forest Classifier using the training dataset.

### 6. Prediction

Generated survival predictions on the testing dataset.

### 7. Evaluation

Evaluated the model using accuracy and classification metrics.

---

## Results

### Model Accuracy

**82.12%**

### Most Important Features

1. Sex
2. Fare
3. Age

These features had the greatest impact on passenger survival prediction.

---

## Files Included

* Titanic-Dataset.csv
* titanic_prediction.py

---

## Conclusion

The Random Forest model successfully learned patterns from historical Titanic passenger data and achieved an accuracy of 82.12% in predicting passenger survival. This project provided practical experience in data preprocessing, machine learning model development, and performance evaluation.

---

## Author

**Anirudh L**

CodSoft Data Science Internship - Task 1
