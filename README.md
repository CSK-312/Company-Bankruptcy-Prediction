# Company Bankruptcy Prediction

A hybrid machine learning framework leveraging combined probabilities from Gaussian Naive Bayes and Deep Neural Networks to predict corporate bankruptcy.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Ensemble and Threshold Tuning](#ensemble-and-threshold-tuning)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)
- [License](#license)

## Project Overview
This project implements a hybrid bankruptcy prediction framework combining a Deep Neural Network (DNN) and Gaussian Naive Bayes (GNB) classifier. It demonstrates:
- Data cleaning and feature engineering
- Handling class imbalance with SMOTE
- ANOVA-based feature selection
- Model ensembling with soft voting and threshold tuning

## Dataset
- **data.csv**: The training dataset containing 5,455 samples and 96 financial features.
- Target variable: `Bankrupt?` (0 = Non-bankrupt, 1 = Bankrupt)

## Methodology
### Exploratory Data Analysis (EDA)
Performed in `Code.ipynb`. Includes:
- Dataset inspection and summary statistics
- Missing value analysis and outlier detection
- Feature correlation analysis to remove highly collinear features

### Data Preprocessing
1. **Feature Selection**: ANOVA F-test to select top 30 features.
2. **Imputation**: Median imputation for missing values.
3. **Standardization**: Scaling features with `StandardScaler`.
4. **SMOTE**: Oversampling minority class on training set only.

### Model Architecture
- **Deep Neural Network (DNN)**: `dnn_model.h5`
  - Input: 30 features
  - Hidden layers: 256 → 128 → 64 neurons with ReLU, batch normalization, and dropout
  - Output: Sigmoid activation for probability
- **Gaussian Naive Bayes**: `GaussianNB_model.pkl`
  - Assumes normal distribution and feature independence

### Ensemble and Threshold Tuning
- Soft voting: Average probabilities from DNN and GNB
- Classification threshold tuned to 0.45 to maximize F1-score

## Results
- Test Accuracy: 97.23%
- F1-Score: 0.51
- Detailed metrics and confusion matrix are available in `Report.pdf`.

## Project Structure
```
.
├── Code.ipynb              # EDA, preprocessing, and model training
├── Evaluation.ipynb        # Testing on new datasets using saved models
├── data.csv                # Training dataset
├── dnn_model.h5            # Trained DNN model
├── scaler.pkl              # Saved StandardScaler instance
├── GaussianNB_model.pkl    # Trained GNB model
├── Report.pdf              # Project report with methodology and results
└── README.md               # Project overview and instructions
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/CSK-312.git
   cd CSK-312
   ```
2. Create a Python environment and install dependencies:
   ```bash
   pip install numpy pandas scikit-learn tensorflow imbalanced-learn matplotlib seaborn
   ```

## Usage
1. **Exploratory Data Analysis & Training**  
   Open `Code.ipynb` in Jupyter Notebook or JupyterLab and run all cells to reproduce the data analysis and model training pipeline.
2. **Evaluation on New Data**  
   Open `Evaluation.ipynb` and follow the instructions to load `dnn_model.h5`, `scaler.pkl`, and `GaussianNB_model.pkl` for predictions on new datasets.
3. **Report**  
   See `Report.pdf` for a detailed write-up of methodology, experiments, and discussions.

## Authors
- Ch. V. Sai Koushik (230312) – skoushik23@iitk.ac.in  
- Srijani Gadupudi (231033) – srijanig23@iitk.ac.in  
- Chilamakuri Kundan Sai (230330) – ckundans23@iitk.ac.in  
- Macha Mohana Harika (230612) – mharika23@iitk.ac.in  
- Challa Kethan (230317) – kethanc23@iitk.ac.in  

## License
This project is licensed under the MIT License.

