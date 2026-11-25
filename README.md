# Home Loan Default Prediction

**PRCP-1006 – Data Science Capstone Project**  
**Domain:** Banking and Financial Services  
**Project Type:** Machine Learning – Classification

## Project Overview

Home Loan Default Prediction is a machine learning classification project that identifies customers who may be at risk of defaulting on their loans.

The project combines customer application details, financial information, credit bureau history, previous loan applications and repayment behaviour. These data are explored and transformed into customer-level features to train and evaluate classification models for credit-risk assessment.

The main application dataset contains **307,511 records and 122 columns**. Its target variable is imbalanced: approximately **8.07%** of applicants are labelled as defaulters.

## Business Problem

Financial institutions need reliable ways to assess the likelihood that a loan applicant may fail to meet repayment obligations. Traditional assessment methods may not capture complex relationships across a customer's financial profile and historical credit behaviour.

This project applies machine learning to historical loan and repayment data to identify potential default patterns and support further credit-risk investigation.

## Project Objectives

- Perform Exploratory Data Analysis (EDA) on customer and financial data.
- Identify factors associated with loan default.
- Analyze missing values, duplicates, anomalies and outliers.
- Engineer customer-level features from historical credit datasets.
- Integrate multiple datasets using customer and loan identifiers.
- Build and compare machine learning classification models.
- Evaluate model performance using metrics suitable for an imbalanced target.
- Optimize the classification threshold and examine feature importance.
- Save the final model and preprocessing pipeline for reuse.

## Dataset Description

The project uses seven CSV datasets:

| Dataset | Description |
|---|---|
| `application_train.csv` | Main loan application data, including customer details and the target variable. |
| `bureau.csv` | Previous credits reported by other financial institutions. |
| `bureau_balance.csv` | Monthly status information for previous credits reported to the credit bureau. |
| `previous_application.csv` | Historical loan applications made by customers. |
| `POS_CASH_balance.csv` | Monthly records of previous point-of-sale and cash loans. |
| `credit_card_balance.csv` | Historical credit card balance and repayment information. |
| `installments_payments.csv` | Scheduled and actual installment payment records. |

**Target variable – `TARGET`:**

- `0` – Non-Defaulter
- `1` – Defaulter

The six supporting datasets provide historical information that is aggregated and joined to the main application data using customer and loan identifiers such as `SK_ID_CURR`, `SK_ID_PREV` and `SK_ID_BUREAU`.

> The CSV datasets are not included in this README. Make sure you have access to all seven files before running the notebook.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project Workflow

1. Load and inspect the main application dataset and supporting datasets.
2. Perform exploratory data analysis on numerical and categorical features.
3. Analyze the target distribution and class imbalance.
4. Investigate missing values, duplicate records and unusual values.
5. Engineer features from bureau, previous application, POS cash, credit card and installment records.
6. Aggregate historical records at customer level and integrate them with the main dataset.
7. Handle anomalies, including the placeholder value in `DAYS_EMPLOYED`.
8. Split the data into training and testing sets using an 80:20 stratified split.
9. Build preprocessing pipelines for missing-value imputation and categorical encoding.
10. Train and evaluate classification models.
11. Optimize the classification threshold using a validation dataset.
12. Compare model performance and calculate permutation feature importance.
13. Save the final model, preprocessing pipeline, feature columns and threshold using Joblib.
14. Load the saved artifact and test the reusable prediction function.

## Machine Learning Models

The notebook evaluates the following classification algorithms:

1. Logistic Regression
2. Random Forest Classifier
3. HistGradientBoosting Classifier

The final saved model is **HistGradientBoosting**. The notebook uses a classification threshold of **0.15** to convert predicted default probabilities into class predictions.

## Final Model Performance

The following results are from the final model evaluation on the test set in the notebook:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 85.35% |
| Precision | 26.60% |
| Recall | 46.34% |
| F1-score | 33.80% |
| ROC-AUC | 0.7793 |
| PR-AUC | 0.2777 |

Because the dataset is imbalanced, accuracy alone does not fully describe model performance. Precision, recall, F1-score and PR-AUC are also considered when assessing the model's ability to identify potential defaulters.

## Feature Importance

Permutation importance using PR-AUC identified the following among the most influential features:

- `EXT_SOURCE_3`
- `EXT_SOURCE_2`
- `AMT_GOODS_PRICE`
- `EXT_SOURCE_1`
- `LATE_PAYMENT_RATE`
- `AMT_ANNUITY`
- `AMT_CREDIT`
- `DAYS_BIRTH`

These features reflect external credit information, loan-related financial details, applicant characteristics and historical repayment behaviour.

## Model Saving and Prediction

The notebook saves the trained HistGradientBoosting model and its fitted preprocessing pipeline using Joblib. The saved artifact also contains the selected threshold and the feature columns required for prediction.

**Saved artifact:** `home_loan_default_model.pkl`

The notebook defines a reusable function named `predict_home_loan_default(customer_data)`. It accepts customer records with the required feature columns and returns:

- `Default_Probability` – predicted probability of default.
- `Predicted_Class` – `0` or `1`, based on the 0.15 threshold.
- `Risk_Category` – `Lower Predicted Default Risk` or `Potential Default Risk`.

The model artifact is generated when the notebook is run through the model-saving section; it is not created by downloading this README alone.

## Repository Structure

A suggested repository organization is:

```text
Home-Loan-Default-Prediction/
│
├── Home Loan Default - Risk Management.ipynb
├── README.md
├── home_loan_default_model.pkl       # Generated by running the notebook
│
└── datasets/
    ├── application_train.csv
    ├── bureau.csv
    ├── bureau_balance.csv
    ├── previous_application.csv
    ├── POS_CASH_balance.csv
    ├── credit_card_balance.csv
    └── installments_payments.csv
```

## How to Run the Project

1. Install Python and Jupyter Notebook.
2. Install the required Python libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn joblib jupyter
   ```

3. Download or otherwise obtain all seven CSV datasets and arrange them in your project directory.
4. Open `Home Loan Default - Risk Management.ipynb` in Jupyter Notebook.
5. The notebook currently reads the CSV files by filename from its working directory. Place the CSV files in that working directory, or update the file paths in the notebook if you keep them inside the `datasets/` folder.
6. Run the notebook cells in order. The final sections train and evaluate the model, save `home_loan_default_model.pkl`, load the saved artifact and demonstrate a prediction.

## Conclusion

This project demonstrates how machine learning can be applied to financial risk analysis by combining loan application information with historical credit and repayment records. The final HistGradientBoosting model produces default-risk probabilities and classifies applicants using a threshold of 0.15.

The model is an experimental decision-support tool. Its predictions should be treated as risk indicators for further review, not as automatic loan approval or rejection decisions. Additional validation, fairness assessment, business-cost analysis, regulatory review and human oversight are necessary before any real-world deployment.

## Author

**K. Prasanth sai**  
Data Science | Machine Learning | Artificial Intelligence
