# Lead Scoring Case Study

## Overview
This project involves a lead scoring analysis using machine learning techniques to predict the likelihood of converting a lead into a customer. The dataset consists of various features related to customer behavior and demographics.

## Table of Contents
- [Dataset](#dataset)
- [Objective](#objective)
- [Approach](#approach)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)


## Dataset
The dataset contains information about leads and their interactions, with features such as:
- Lead Source
- Total Time Spent on Website
- Lead Quality
- Converted (Target Variable)
- Many more........

## Objective
The goal is to develop a predictive model that can assign lead scores based on historical data, helping businesses prioritize leads effectively.

## Approach
1. Data Preprocessing:
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling
2. Model Selection:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbors (KNN)
3. Model Evaluation:
   - Accuracy
   - Precision, Recall, F1-score
   - ROC-AUC Curve

## Model Evaluation
The Random Forest model is evaluated using performance metrics, and the best model is selected based on its predictive power. Classification is performed using the Random Forest model.

## Results
The best-performing model achieves a high accuracy and AUC score, providing reliable lead scoring predictions.
- Training Sensitivity: 92.88%
- Training Specificity: 98.36%
- Training Accuracy: 96.25%
- Testing Sensitivity: 88.98%
- Testing Specificity: 96.31%
- Testing Accuracy: 93.63%

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/lead-scoring.git
cd lead-scoring
pip install -r requirements.txt
```

## Usage
Run the Jupyter Notebook to train and evaluate the models:
```bash
jupyter notebook lead_scoring_case_study.ipynb
```
Modify hyperparameters and model selection in the notebook as needed.

## Dependencies
This project requires:
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

Install dependencies using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```




