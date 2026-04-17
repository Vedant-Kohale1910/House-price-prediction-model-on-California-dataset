🏠 California Housing Price Prediction

This project is a Machine Learning pipeline built to predict housing prices in California using the California Housing Dataset. It demonstrates an end-to-end workflow including data preprocessing, feature engineering, model training, and inference.

📌 Project Overview

The goal of this project is to build a robust regression model that predicts the median house value based on various socio-economic and geographical features.

This implementation follows production-level practices:

Stratified sampling
Data preprocessing pipelines
Model persistence using joblib
Separate training and inference workflows
📂 Project Structure
├── main.py              # Production pipeline (training + inference)
├── main_old.py         # Experimental model comparisons
├── housing_data.csv    # Dataset (ignored via .gitignore)
├── model.pkl           # Saved trained model
├── pipeline.pkl        # Saved preprocessing pipeline
├── input.csv           # Input data for inference
├── output.csv          # Predictions output
└── README.md
⚙️ Features & Workflow
1. Data Preprocessing
Handling missing values using Median Imputation
Feature scaling using StandardScaler
Categorical encoding using OneHotEncoder
Combined using ColumnTransformer + Pipeline
2. Stratified Sampling
Dataset is split using StratifiedShuffleSplit
Based on median income categories
Ensures balanced representation across income groups
3. Model Training
Final model used: Random Forest Regressor
Trained on fully processed pipeline output
4. Model Persistence
Model saved as: model.pkl
Pipeline saved as: pipeline.pkl
Enables reuse without retraining
5. Inference Pipeline
Loads saved model & pipeline
Transforms new input data
Generates predictions
Saves results to output.csv
🚀 How to Run
Step 1: Install Dependencies
pip install pandas numpy scikit-learn joblib
Step 2: Train the Model

Make sure housing_data.csv is present.

python main.py

Output:

model is trained!
Step 3: Run Inference

Run the same command again:

python main.py

Output:

Inference is completed, Results saved to output.csv
📊 Models Compared (main_old.py)

The following models were evaluated:

Linear Regression
Decision Tree Regressor
Random Forest Regressor ✅ (Selected)

Evaluation Metric:

Root Mean Squared Error (RMSE)
Cross-validation used for reliability
📈 Key Concepts Demonstrated
End-to-End ML Pipeline
Feature Engineering
Model Evaluation & Selection
Pipeline Serialization
Production-ready Code Structure
🧠 Future Improvements
Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
Model versioning
API deployment (Flask / FastAPI)
CI/CD integration
Feature importance visualization
📚 Dataset
California Housing Dataset
Contains features like:
Median Income
House Age
Total Rooms
Population
Ocean Proximity
