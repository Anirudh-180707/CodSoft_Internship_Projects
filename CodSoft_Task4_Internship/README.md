# Sales Prediction Using Python

## Overview

Sales prediction is the process of forecasting future sales based on various factors such as advertising expenditure. In this project, a Machine Learning model is built using Linear Regression to predict product sales based on advertising budgets allocated to TV, Radio, and Newspaper marketing channels.

This project demonstrates how data science techniques can help businesses make informed decisions regarding advertising strategies and improve sales performance.

---

## Dataset

The dataset contains advertising expenditures across different media platforms along with corresponding sales figures.

### Features

- **TV** – Advertising budget spent on TV
- **Radio** – Advertising budget spent on Radio
- **Newspaper** – Advertising budget spent on Newspaper
- **Sales** – Product sales (Target Variable)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

---

## Machine Learning Algorithm

### Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values. It identifies the relationship between advertising expenditures and product sales.

---

## Project Workflow

1. Load and explore the dataset.
2. Select input features and target variable.
3. Split the dataset into training and testing sets.
4. Train a Linear Regression model.
5. Evaluate model performance using various metrics.
6. Predict sales based on user-provided advertising budgets.

---

## Evaluation Metrics

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Sales_Prediction.git
```

Navigate to the project directory:

```bash
cd Sales_Prediction
```

Install required libraries:

```bash
pip install pandas numpy scikit-learn
```

---

## Running the Project

Run the Python file:

```bash
python Sales_Prediction.py
```

---

## Sample Input

```text
Enter TV Advertising Budget: 230
Enter Radio Advertising Budget: 38
Enter Newspaper Advertising Budget: 69
```

### Sample Output

```text
Predicted Sales: 22.12
```

---

## Results

The model successfully predicts sales based on advertising expenditures and achieves a high R² score, indicating strong predictive performance.

---

## Learning Outcomes

Through this project, I learned:

- Data preprocessing using Pandas
- Feature selection
- Train-test splitting
- Building a Linear Regression model
- Model evaluation techniques
- Making predictions using machine learning

---

## Future Enhancements

- Use advanced regression algorithms
- Add data visualization
- Build an interactive web application
- Deploy the model for real-time predictions

---

## Author

Anirudh L

GitHub: https://github.com/Anirudh-180707
