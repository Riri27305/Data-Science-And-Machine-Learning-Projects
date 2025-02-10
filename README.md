# Projects-and-experience
# Heart Disease Prediction using Machine Learning

## Overview

Heart disease is one of the leading causes of mortality worldwide. This project leverages machine learning techniques to predict the likelihood of heart disease based on clinical and demographic attributes. By analyzing a dataset of patient records, the model aims to assist healthcare professionals in making data-driven decisions for early diagnosis and prevention.

## Features

- Exploratory Data Analysis (EDA) for insights into the dataset
- Data preprocessing including feature scaling and encoding
- Multiple machine learning models for classification:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Neural Networks
- Performance evaluation using accuracy, precision, recall, F1-score, and ROC curve
- Model interpretability and feature importance analysis

## Dataset

The dataset (`heart.csv`) contains 1,025 patient records with 14 key features, including age, cholesterol levels, blood pressure, and ECG results. The target variable indicates the presence (1) or absence (0) of heart disease.

## Installation

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Clone Repository

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### Running the Project

1. Open `Project_10_Heart_Disease_Prediction.ipynb` in Google Colab or Jupyter Notebook.
2. Load the dataset and execute all cells.
3. Compare the model performances and analyze predictions.

## Usage

- Modify hyperparameters to optimize model performance.
- Experiment with additional datasets for better generalization.
- Deploy the model using Flask or FastAPI for real-time predictions.

## Results

- The **Random Forest model** achieved the highest accuracy.
- **Key influencing factors** include chest pain type, cholesterol, and exercise-induced angina.
- The model provides a valuable tool for early heart disease detection.

## Future Improvements

- Implement deep learning techniques for enhanced accuracy.
- Integrate real-time patient data from IoT-enabled devices.
- Deploy as a web-based or mobile application for accessibility.

## Contributors

- **Riya Garg**





