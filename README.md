# 🩺 Diabetes Prediction Using Machine Learning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-BRFSS_2015-red.svg)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

**A comprehensive Machine Learning project for predicting diabetes risk using health indicators**

[📊 View Report](report/Diabetes%20Prediction%20Using%20Machine%20Learning.pdf) • [📁 Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) • [🔬 Notebooks](notebooks/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset Information](#-dataset-information)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Models & Results](#-models--results)
- [Team](#-team)
- [References](#-references)

---

## 🎯 Overview

This project implements and compares **5 traditional Machine Learning models** to predict diabetes risk using the **BRFSS 2015 Health Indicators Dataset** containing **253,680+ samples** and **21 health features**.

### 🎓 Academic Context

- **Course:** Intelligent Agents - Machine Learning
- **Institution:** Faculty of Computers & Artificial Intelligence, Benha University
- **Department:** Information Systems - Section 3
- **Instructor:** [Eng. Yousef El-Baroudy](https://github.com/YousefTB)
- **Supervisor:** Dr. Fady Mohamed

---

## ✨ Key Features

- ✅ **Comprehensive Data Pipeline:** Complete data cleaning, preprocessing, and feature engineering
- ✅ **5 ML Models Implemented:** KNN, Decision Tree, Naive Bayes, Random Forest, SVM
- ✅ **In-depth EDA:** Correlation analysis, chi-square tests, feature importance
- ✅ **Performance Comparison:** Detailed metrics (Accuracy, F1-Score, Precision, Recall)
- ✅ **Class Imbalance Handling:** Analysis of imbalanced dataset challenges
- ✅ **Professional Documentation:** Full technical report with visualizations

---

## 📊 Dataset Information

**Source:** [Diabetes Health Indicators Dataset - Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

| Metric | Value |
|--------|-------|
| **Total Samples** | 253,680 |
| **After Cleaning** | 229,781 |
| **Features** | 21 health indicators |
| **Target Classes** | 3 (No Diabetes, Pre-Diabetes, Diabetes) |
| **Class Distribution** | 82% / 2% / 15% (Highly Imbalanced) |

### Features Include:
- Demographics: Age, Sex, Education, Income
- Health Indicators: BMI, Blood Pressure, Cholesterol
- Lifestyle: Physical Activity, Smoking, Diet
- Medical History: Heart Disease, Stroke

---

## 📁 Project Structure

```
ML-Diabetes-Project/
│
├── 📂 data/
│   ├── raw/                          # Original datasets from Kaggle
│   │   ├── diabetes_012_health_indicators_BRFSS2015.csv
│   │   ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│   │   └── diabetes_binary_health_indicators_BRFSS2015.csv
│   │
│   └── processed/                    # Cleaned & preprocessed data
│       ├── cleaned_dataset_full.csv  # Cleaned dataset
│       ├── scaler.pkl                # StandardScaler object
│       ├── X_train.csv               # Training features
│       ├── X_test.csv                # Testing features
│       ├── y_train.csv               # Training labels
│       └── y_test.csv                # Testing labels
│
├── 📂 notebooks/                     # Jupyter notebooks (run in order)
│   ├── 1️⃣ data_preprocessing.ipynb  # Data cleaning & preprocessing
│   ├── 2️⃣ eda_analysis.ipynb        # Exploratory Data Analysis
│   └── 3️⃣ traditional_models.ipynb  # Model training & evaluation
│
├── 📂 models/                        # Saved trained models
│   ├── .gitkeep                      # (Folder placeholder)
│   ├── decision_tree_model.pkl       # ⚠️ Generated after running notebook 3
│   ├── knn_model.pkl                 # ⚠️ Generated after running notebook 3
│   ├── naive_bayes_model.pkl         # ⚠️ Generated after running notebook 3
│   ├── random_forest_model.pkl       # ⚠️ Generated after running notebook 3
│   └── svm_linear_model.pkl          # ⚠️ Generated after running notebook 3
│
├── 📂 report/
│   └── Diabetes Prediction Using Machine Learning.pdf  # Full technical report
│
├── 📄 .gitignore
├── 📄 README.md
└── 📄 requirements.txt               # Python dependencies
```

> **⚠️ Important:** The `models/` folder is **empty** in the repository. Model files (`.pkl`) are generated locally after running `traditional_models.ipynb`.

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Jupyter Notebook / JupyterLab
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/MuhammadAbdelkader/ML-Diabetes-Project.git
cd ML-Diabetes-Project
```

### Step 2: Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

1. Go to [Kaggle Dataset Page](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
2. Download the dataset (3 CSV files)
3. Place all CSV files in `data/raw/` folder

---

## 📖 Usage Guide

### Running the Project (Step by Step)

Open Jupyter Notebook:
```bash
jupyter notebook
```

Then execute the notebooks **in this order:**

#### 1️⃣ Data Preprocessing (`data_preprocessing.ipynb`)

**What it does:**
- Loads raw dataset
- Removes duplicates (23,899 rows)
- Applies StandardScaler normalization
- Splits data into train/test (80/20 with stratification)
- Saves processed data to `data/processed/`

**Output:**
- `cleaned_dataset_full.csv`
- `X_train.csv`, `X_test.csv`
- `y_train.csv`, `y_test.csv`
- `scaler.pkl`

---

#### 2️⃣ Exploratory Data Analysis (`eda_analysis.ipynb`)

**What it does:**
- Correlation analysis (identifies top features)
- Chi-square tests for categorical features
- BMI & Age distribution analysis
- Lifestyle factors impact study
- Visualization of key patterns

**Key Findings:**
- Top correlated features: General Health (0.283), High BP (0.260), BMI (0.210)
- Severe class imbalance: 82% no diabetes, 2% pre-diabetes, 15% diabetes
- BMI and Age strongly correlated with diabetes risk

---

#### 3️⃣ Traditional Models (`traditional_models.ipynb`)

**What it does:**
- Trains 5 ML models: KNN, Decision Tree, Naive Bayes, Random Forest, SVM
- Evaluates using: Accuracy, Precision, Recall, F1-Score
- Generates confusion matrices
- Saves trained models to `models/` folder

**⚠️ Important:** This notebook **creates the model files** in `models/` folder:
- `decision_tree_model.pkl`
- `knn_model.pkl`
- `naive_bayes_model.pkl`
- `random_forest_model.pkl`
- `svm_linear_model.pkl`

---

## 🏆 Models & Results

### Model Comparison

| Model | Accuracy | F1-Score | Training Speed | Best For |
|-------|----------|----------|----------------|----------|
| **SVM Linear** | **83.24%** | 77.41% | Fast ⚡ | Overall accuracy |
| **Random Forest** | 82.37% | **78.66%** | Medium 🔄 | Balanced performance |
| **KNN** | 81.56% | 78.37% | Slow 🐢 | Small datasets |
| **Naive Bayes** | 74.14% | 75.95% | Very Fast ⚡⚡ | Detecting diabetics (Class 2) |
| **Decision Tree** | 74.18% | 74.82% | Fast ⚡ | Interpretability |

### Per-Class Performance (F1-Scores)

| Model | Class 0 (No Diabetes) | Class 1 (Pre-Diabetes) | Class 2 (Diabetes) |
|-------|----------------------|------------------------|---------------------|
| SVM Linear | 0.91 | **0.00** ❌ | 0.15 |
| Random Forest | 0.90 | **0.00** ❌ | 0.27 |
| KNN | 0.90 | **0.00** ❌ | 0.28 |
| **Naive Bayes** | 0.84 | 0.02 | **0.42** ✅ |
| Decision Tree | 0.85 | 0.03 | 0.30 |

### Key Insights

✅ **Best Overall Model:** SVM Linear (83.24% accuracy)
✅ **Most Balanced Model:** Random Forest (F1=78.66%)
✅ **Best for Detecting Diabetes:** Naive Bayes (F1=0.42 for Class 2)
❌ **Critical Challenge:** All models failed on Class 1 (Pre-Diabetes) due to severe class imbalance (only 2% of data)

---

## 📈 Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | BMI | 18.30% |
| 2 | Age | 12.33% |
| 3 | Income | 10.24% |
| 4 | PhysHlth | 8.44% |
| 5 | Education | 7.33% |

---

## 🔮 Future Improvements

- 🎯 Apply **SMOTE** (Synthetic Minority Over-sampling) to handle class imbalance
- ⚙️ Implement **GridSearchCV** for hyperparameter tuning
- 🌳 Test **XGBoost** and **LightGBM** models
- 🔄 Use **5-fold Cross-Validation** instead of single split
- 🧬 Create **interaction features** (e.g., BMI × Age)

---

## 👥 Team

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/MohammedAshraF0">
        <img src="https://github.com/MohammedAshraF0.png" width="100px;" alt="Mohamed Ashraf"/><br />
        <sub><b>Mohamed Ashraf</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/MuhammadAbdelkader">
        <img src="https://github.com/MuhammadAbdelkader.png" width="100px;" alt="Mohamed Abdelkader"/><br />
        <sub><b>Mohamed Abdelkader</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/m7medfathy">
        <img src="https://github.com/m7medfathy.png" width="100px;" alt="Mohamed Fathy"/><br />
        <sub><b>Mohamed Fathy</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/mohamedosama2004">
        <img src="https://github.com/mohamedosama2004.png" width="100px;" alt="Mohamed Osama"/><br />
        <sub><b>Mohamed Osama</b></sub>
      </a>
    </td>
  </tr>
</table>

### Supervision

- **Instructor:** [Eng. Yousef El-Baroudy](https://github.com/YousefTB)
- **Supervisor:** Dr. Fady Mohamed

---

## 📚 References

1. [Diabetes Health Indicators Dataset - Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
2. [BRFSS 2015 - CDC](https://www.cdc.gov/brfss/)
3. [Scikit-learn Documentation](https://scikit-learn.org/)
4. [Handling Imbalanced Datasets - SMOTE](https://imbalanced-learn.org/)

---

## 📄 License

This project is created for **academic purposes** as part of the Intelligent Agents course at Benha University.

---

## 🤝 Contributing

This is an academic project and is not open for contributions. However, you can:
- ⭐ Star the repository if you find it useful
- 🐛 Report issues
- 💡 Suggest improvements via issues

---

## 📧 Contact

For questions or discussions about this project:

- Open an issue on GitHub
- Contact the team via university email

---

<div align="center">

**Made with ❤️ by Team Information Systems**

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

</div>
