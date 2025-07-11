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

---

# Movie Recommendation System (MRS)

This project is a **Movie Recommendation System** built using Python and popular data science libraries. It provides personalized movie recommendations to users based on collaborative filtering and similarity metrics.

## 📌 Project Overview

The Movie Recommendation System (MRS) demonstrates the implementation of a basic content-based filtering method to recommend movies. The system analyzes user preferences and movie metadata to suggest films that align with user interests.

## 🔍 Features

- Content-based filtering using cosine similarity
- Data preprocessing and cleaning using pandas and NumPy
- Interactive user input for generating recommendations
- Uses the TMDB 5000 Movie Dataset
- Implements feature extraction on genres, keywords, director, and cast
- Cosine similarity based recommendation logic

## 📁 Dataset

The system uses the following datasets:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

These datasets can be downloaded from [Kaggle TMDB 5000 Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

## 🛠️ Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Jupyter Notebook

## 🚀 Getting Started

### Prerequisites

Make sure you have the following libraries installed:

pip install pandas numpy scikit-learn nltk

🎯 How It Works:
Preprocess movie metadata to extract relevant features

Combine features into a single "tags" column

Convert textual data into vectors using CountVectorizer

Compute similarity matrix using cosine similarity

Recommend top N similar movies based on user input

📌 Sample Output:
If the user inputs:

Input movie: Avatar

The system might return:

Recommended movies:
1. John Carter
2. Guardians of the Galaxy
3. Prometheus
4. ...

🧠 Future Improvements:
Implement hybrid filtering (collaborative + content-based)

Integrate with a web framework (e.g., Flask or Streamlit)

Improve feature engineering using advanced NLP techniques


Feel free to fork this project and contribute!


---





