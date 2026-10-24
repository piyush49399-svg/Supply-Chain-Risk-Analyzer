# Supply Chain Risk Analyzer

An interactive data analytics and machine learning project designed to analyze supply chain delivery performance, identify potentially high-risk orders, and explore factors associated with delivery delays.

## Project Overview

Late deliveries can affect customer satisfaction, operational efficiency, and supply chain costs. This project uses historical supply chain data to analyze delivery performance and help users identify orders that may require attention.

The application combines exploratory data analysis, machine learning, and an interactive Streamlit dashboard.

## Key Features

- **Interactive Dashboard:** Explore supply chain and delivery performance.
- **Shipping Performance Analysis:** Compare delivery risk across shipping modes.
- **Delivery Delay Analysis:** Analyze early, on-time, and late deliveries.
- **Order Risk Identification:** Display orders flagged for attention.
- **Machine Learning:** Train and evaluate models for delivery risk classification.
- **Model Comparison:** Compare the performance of candidate models.
- **Feature Importance:** Examine which features contribute most to model decisions.
- **Early Risk Estimation:** Estimate delivery risk from selected order details using a trained model.
- **CSV Export:** Download orders flagged for attention.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Joblib

## Project Structure

```text
Supply-Chain-Risk-Analyze/
├── data/
├── models/
├── outputs/
├── app.py
├── Dashboard.py
├── train_model.py
├── evaluate_model.py
├── compare_models.py
├── early_risk_model.py
├── feature_importance.py
├── validate_data.py
├── confusion_matrix.png
├── model_comparison.csv
├── model_comparison_all.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Supply-Chain-Risk-Analyze
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the application

To launch the dashboard:

```bash
python -m streamlit run Dashboard.py
```

To launch the alternative application, if required:

```bash
python -m streamlit run app.py
```

## Machine Learning Workflow

1. Validate the dataset.
2. Prepare the data for analysis and modeling.
3. Train candidate machine learning models.
4. Evaluate and compare model performance.
5. Analyze feature importance.
6. Integrate the trained models into the application.
7. Display delivery risk estimates and relevant analytical insights.

## Model Evaluation

The project includes model comparison results and a confusion matrix.

Actual evaluation metrics should be documented here after reviewing the generated results. Model performance should be assessed using appropriate classification metrics rather than accuracy alone.

## Limitations

- Historical delivery patterns may not represent future operating conditions.
- Model estimates depend on the quality and relevance of the input data.
- Estimated risk is not a guarantee that an order will be delivered late.
- Performance should be validated on appropriate unseen data before real-world operational use.

## Future Improvements

- Deploy the dashboard online.
- Add monitoring for changes in model performance.
- Improve prediction reliability through further validation and feature engineering.
- Add more detailed operational insights and interactive filters.

## Author

Piyush Sharma

Computer Science Engineering — Data Science