# Movie Rating Prediction Using Machine Learning

## Project Overview

This project focuses on predicting movie ratings using Machine Learning techniques. By analyzing movie-related information such as genre, director, actors, duration, release year, and vote count, the model estimates the rating a movie is likely to receive.

The project demonstrates data preprocessing, feature engineering, model training, and evaluation using a regression algorithm.

---

## Objective

To build a Machine Learning model that predicts movie ratings based on various movie attributes and historical data.

---

## Dataset Information

The dataset contains information about Indian movies, including:

* Movie Name
* Release Year
* Duration
* Genre
* Rating
* Votes
* Director
* Actor 1
* Actor 2
* Actor 3

### Target Variable

**Rating** – The movie rating to be predicted.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn

---

## Machine Learning Algorithm

### Random Forest Regressor

Random Forest Regressor is an ensemble learning algorithm that combines multiple decision trees to predict continuous numerical values with improved accuracy.

---

## Project Workflow

### 1. Data Loading

Loaded the movie dataset using Pandas.

### 2. Data Preprocessing

* Removed missing values
* Cleaned Year, Duration, and Votes columns
* Converted categorical data into numerical format using Label Encoding

### 3. Feature Selection

Selected the following features:

* Year
* Duration
* Votes
* Genre
* Director
* Actor 1
* Actor 2
* Actor 3

### 4. Train-Test Split

Split the dataset into:

* 80% Training Data
* 20% Testing Data

### 5. Model Training

Trained a Random Forest Regressor on the training dataset.

### 6. Prediction

Generated movie rating predictions using the testing dataset.

### 7. Evaluation

Evaluated model performance using:

* Mean Absolute Error (MAE)
* R² Score

---

## Results

### Model Performance

* Mean Absolute Error (MAE): **0.85**
* R² Score: **0.31**

### Feature Importance

The most influential features for predicting movie ratings were:

1. Votes
2. Year
3. Genre
4. Duration
5. Actor 1
6. Actor 2
7. Actor 3
8. Director

---

## Files Included

* IMDb Movies India.csv
* movie_rating_prediction.py
* README.md

---

## Conclusion

The Random Forest Regressor successfully analyzed movie-related features and predicted movie ratings. This project provided practical experience in data preprocessing, regression modeling, feature engineering, and performance evaluation using Python and Machine Learning.

---

## Author

**Anirudh L**

CodSoft Data Science Internship - Task 2
