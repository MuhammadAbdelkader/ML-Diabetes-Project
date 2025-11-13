# ML-Diabetes-Project

## Overview
This project aims to predict diabetes health indicators using a large dataset of health survey data. The project fulfills the academic requirements for the Intelligent Agents course by implementing comprehensive data cleaning, processing, analysis, and applying multiple Machine Learning models to compare their performance.

**Dataset:** [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
**Total Samples:** 253,680+
**Features:** Demographics, lifestyle, health indicators
**Target:** Diabetes diagnosis (Yes/No)

## Team Members

- [Mohamed Ashraf](https://github.com/MohammedAshraF0)
- [Mohamed Fathy](https://github.com/m7medfathy)
- [Mohamed Osama](https://github.com/mohamedosama2004)
- [Mohamed Abdelkader](https://github.com/MuhammadAbdelkader)

---

## Project Goals
- Build a machine learning pipeline to predict diabetes risk
- Clean, process, and analyze the dataset for insights
- Apply multiple Machine Learning models (Traditional + Deep Learning)
- Compare model performances and explain differences
- Document all steps, results, and insights in a comprehensive report

---

## Project Structure

```
ML-Diabetes-Project/
├── data/
│   ├── raw/                    # Original dataset files from Kaggle
│   └── processed/              # Cleaned and processed dataset
├── notebooks/                  # Jupyter Notebooks for EDA and analysis
├── src/                        # Python scripts
│   ├── preprocessing.py
│   ├── traditional_models.py
│   └── deep_learning_models.py
├── models/                     # Saved trained models (pickle/h5)
├── report/                     # Final report and supporting documents
├── README.md
└── requirements.txt            # Required Python libraries
```

---

## Project Workflow

1. **Download Dataset**
   - Download CSV files from Kaggle and place in `data/raw/`

2. **Data Preprocessing**
   - Clean data: remove missing values, normalize, encode categorical features
   - Save processed dataset in `data/processed/`

3. **Exploratory Data Analysis (EDA)**
   - Analyze distribution of features and target
   - Visualize correlations, class balance, and important patterns

4. **Model Training & Evaluation**
   - **Traditional ML Models:** Logistic Regression, SVM, Random Forest, Naive Bayes
   - **Deep Learning Models:** Dense Neural Network (optionally LSTM)
   - Evaluate using metrics: Accuracy, F1-score, Precision, Recall
   - Visualize results using bar charts and confusion matrices

5. **Comparison & Analysis**
   - Compare all models
   - Explain differences in performance, effect of preprocessing, and feature importance

6. **Reporting**
   - Document all steps in `report/`
   - Include data cleaning steps, model hyperparameters, evaluation results, charts, and analysis

---

## Technical Setup

### Required Libraries

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow
keras
jupyter
```

### Installation & Running

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MuhammadAbdelkader/ML-Diabetes-Project.git
   cd ML-Diabetes-Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run preprocessing:**
   ```bash
   python src/preprocessing.py
   ```

4. **Run traditional models:**
   ```bash
   python src/traditional_models.py
   ```

5. **Run deep learning models:**
   ```bash
   python src/deep_learning_models.py
   ```

---

## References
- [Diabetes Health Indicators Dataset on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---

## License
This project is created for academic purposes as part of the Intelligent Agents course.
