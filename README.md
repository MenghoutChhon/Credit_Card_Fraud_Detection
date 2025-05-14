# 🎴 Credit Card Fraud Detection using Machine Learning

[![Watch the video]](https://www.youtube.com/watch?v=kXqvSxNXBOA)

## 📌 Project Overview

This project implements machine learning techniques to detect fraudulent credit card transactions. The system analyzes transaction patterns to identify potentially fraudulent activities with high accuracy. The solution includes both a **Jupyter Notebook** for model development and a **Streamlit web application** for interactive fraud detection.

---

## ✨ Key Features

- **Data Analysis**: Comprehensive exploration of credit card transaction data  
- **Machine Learning Models**: Implementation of multiple algorithms including:
  - Logistic Regression
  - Random Forest
  - Neural Networks (MLP Classifier)
- **Handling Class Imbalance**: Use of **SMOTE technique** to address the highly imbalanced dataset  
- **Model Evaluation**: Detailed performance metrics including accuracy, classification reports, and confusion matrices  
- **Streamlit Interface**: User-friendly web application for real-time fraud detection  

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset from Kaggle**, containing:

- **284,807 transactions**
- **31 features** (28 principal components, `Time`, `Amount`, and `Class`)
- **Highly imbalanced classes** (492 frauds out of 284,807 transactions)

---

## 🛠️ Technologies Used

- Python  
- Scikit-learn  
- Streamlit  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Imbalanced-learn (SMOTE)  

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+  
- Jupyter Notebook  
- Streamlit  

### Installation

Clone the repository:

```bash
git clone https://github.com/MenghoutChhon/credit-card-fraud-detection.git
cd credit-card-fraud-detection
