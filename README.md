# Employee Performance Analysis

## Project Overview
This project aims to analyze and predict employee performance based on various demographic, work, and satisfaction-related features. The insights and predictions are intended to guide the company in identifying key factors that drive performance, thus aiding in hiring, employee management, and overall organizational improvements.

---

## Table of Contents
1. [Business Case](#business-case)
2. [Dataset Overview](#dataset-overview)
3. [Project Goals](#project-goals)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Data Preprocessing](#data-preprocessing)
6. [Feature Selection](#feature-selection)
7. [Machine Learning Models](#machine-learning-models)
8. [Recommendations](#recommendations)
9. [Tools and Libraries](#tools-and-libraries)
10. [How to Use the Project](#how-to-use-the-project)

---

## Business Case
The primary objective of this project is to help the organization predict employee performance ratings and identify the most impactful factors affecting performance. By understanding these factors, the company can improve employee satisfaction, retention, and productivity.

---

## Dataset Overview
- **Total Rows**: 1200
- **Total Columns**: 28 (19 quantitative, 8 qualitative, and 1 alphanumeric identifier)
- **Target Variable**: `PerformanceRating`

### Feature Breakdown
- **Quantitative Features**: 11 numeric, 8 ordinal
- **Qualitative Features**: 8 categorical
- **ID Feature**: `EmpNumber` (unique alphanumeric identifier that is not relevant to performance prediction)

---

## Project Goals
1. **Department-Wise Performance Analysis**: Understand performance distribution across different departments.
2. **Top Factors Affecting Performance**: Identify key features that influence employee performance.
3. **Predictive Model Development**: Build a model to predict employee performance ratings.
4. **Actionable Recommendations**: Provide data-driven recommendations to improve employee performance.

---

## Exploratory Data Analysis
The EDA phase involved investigating the relationships between features and identifying trends.

### Techniques Used:
- **Univariate Analysis**: Explored individual feature distributions and unique labels.
- **Bivariate Analysis**: Analyzed relationships between features and the target variable.
- **Multivariate Analysis**: Examined interactions between multiple features and the target variable.

### Key Insights:
- **Sales Department**: Most employees are rated at level 3 performance, with a slight male advantage.
- **Human Resources**: Female employees tend to perform better, with older employees generally performing lower.
- **Data Science**: High overall performance, especially for male employees.

### Plots Used:
- **Violin Plot**: For distribution comparison across departments.
- **Count Plot**: For categorical feature distributions.
- **Heatmap**: To analyze feature correlation.

---

## Data Preprocessing
Data preprocessing involved preparing the data for modeling by handling missing values, encoding categorical features, and transforming skewed features.

### Steps:
1. **Missing Value Check**: No missing values were found in the dataset.
2. **Encoding**: Used frequency encoding and manual mapping to convert categorical data into numerical values.
3. **Outlier Handling**: Addressed outliers using the IQR method.
4. **Feature Transformation**: Applied square root transformation to reduce skewness in the `YearsSinceLastPromotion` feature.
5. **Scaling**: Standard scaling to normalize features for machine learning.

---

## Feature Selection
Feature selection helped to reduce dimensionality while retaining essential information for prediction.

### Steps:
1. **Dropping Irrelevant Features**: Removed constant and unique features such as `EmpNumber`.
2. **Correlation Analysis**: Used heatmap analysis to understand feature relationships with the target variable.
3. **Principal Component Analysis (PCA)**: Reduced dimensionality from 27 to 25 features, capturing essential variance.

---

## Machine Learning Models
The project used multiple classification algorithms to create a robust model for predicting employee performance.

### Models Used:
1. **Support Vector Classifier**: Achieved 98.28% accuracy.
2. **Random Forest Classifier**: Achieved 95.61% accuracy.
3. **Artificial Neural Network (MLP)**: Achieved 95.80% accuracy.

### Model Selection:
After evaluating performance metrics, the **Artificial Neural Network (MLP)** model was selected as it balanced accuracy and generalization best.

### Data Balancing:
Used **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset, ensuring even representation of performance ratings in the training data.

---

## Recommendations
The analysis yielded several actionable insights to improve employee performance across the organization:

1. **Enhance Work Environment**: Focusing on employee environment satisfaction can significantly impact performance.
2. **Regular Salary Increases**: Implement structured salary hikes to incentivize high performance.
3. **Promotions**: Offer promotions every 6 months to boost motivation.
4. **Work-Life Balance Initiatives**: Improve work-life balance programs to help employees manage their personal and professional commitments.
5. **Female Recruitment in HR**: Female employees tend to perform better in HR roles, offering a focus area for recruitment.

### Department-Specific Insights:
- **Sales & Development**: These departments have higher overall performance; maintaining satisfaction here is critical.
- **Data Science**: The department demonstrates high performance; leveraging this success can benefit other areas.

---

## Tools and Libraries
### Tools
- **Jupyter Notebook**: For data analysis and model building.

### Libraries
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computation.
- **Matplotlib & Seaborn**: Data visualization.
- **SciPy**: Statistical analysis.
- **Scikit-Learn**: Machine learning algorithms.
- **Pickle**: Model saving and loading.

---

## How to Use the Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Tahahussain53110/employee_performance_analysis
   cd employee_performance_analysis
   ```

2. **Data Preparation**:
   - Ensure you have the dataset file in the working directory.

3. **Install Dependencies**:
   Install required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebook**:
   - Open `Employee_Performance_Analysis.ipynb` in Jupyter Notebook.
   - Execute each cell to preprocess the data, analyze it, and train models.

5. **Use the Trained Model**:
   - The trained model is saved using Pickle. Load it using:
   ```python
   import pickle
   with open('model.pkl', 'rb') as file:
       model = pickle.load(file)
   ```
   - Use the model to predict employee performance by providing input data.

---

## Conclusion
This project provides a comprehensive analysis and prediction model for employee performance based on key factors. The insights gained can be used to drive strategic decisions in employee management and retention, ultimately contributing to improved organizational performance.

For further details, please check the [GitHub Repository](https://github.com/Tahahussain53110/employee_performance_analysis).