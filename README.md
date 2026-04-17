# 🏠 California Housing Price Prediction

This project is a **Machine Learning pipeline** built to predict housing prices in California using the **California Housing Dataset**. It demonstrates an end-to-end workflow including data preprocessing, feature engineering, model training, and inference.

---

## 📌 Project Overview

The goal of this project is to build a regression model that predicts the **median house value** based on socio-economic and geographical features.

This project follows **production-level practices**:

* Stratified sampling
* Data preprocessing pipelines
* Model persistence using `joblib`
* Separate training and inference workflows

---

## 📂 Project Structure

```
├── main.py              # Production pipeline (training + inference)
├── main_old.py         # Experimental model comparisons
├── housing_data.csv    # Dataset (ignored via .gitignore)
├── model.pkl           # Saved trained model
├── pipeline.pkl        # Saved preprocessing pipeline
├── input.csv           # Input data for inference
├── output.csv          # Predictions output
└── README.md
```

---

## ⚙️ Features & Workflow

### 1. Data Preprocessing

* Missing values handled using **Median Imputation**
* Feature scaling using **StandardScaler**
* Categorical encoding using **OneHotEncoder**
* Combined using **ColumnTransformer + Pipeline**

---

### 2. Stratified Sampling

* Implemented using **StratifiedShuffleSplit**
* Based on **median income categories**
* Ensures balanced dataset distribution

---

### 3. Model Training

* Final model: **Random Forest Regressor**
* Trained on processed pipeline data

---

### 4. Model Persistence

* Model saved as `model.pkl`
* Pipeline saved as `pipeline.pkl`
* Avoids retraining for inference

---

### 5. Inference Pipeline

* Loads saved model and pipeline
* Transforms input data
* Generates predictions
* Saves results to `output.csv`

---

## 🚀 How to Run

### 1. Install Dependencies

```
pip install pandas numpy scikit-learn joblib
```

---

### 2. Train the Model

Make sure `housing_data.csv` is present:

```
python main.py
```

Output:

```
model is trained!
```

---

### 3. Run Inference

Run the same command again:

```
python main.py
```

Output:

```
Inference is completed, Results saved to output.csv
```

---

## 📊 Models Compared (main_old.py)

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor ✅ (Selected)

**Evaluation Metric:**

* Root Mean Squared Error (RMSE)
* Cross-validation

---

## 📈 Key Concepts Demonstrated

* End-to-End ML Pipeline
* Feature Engineering
* Model Evaluation
* Pipeline Serialization
* Production-ready structure

---

## 🧠 Future Improvements

* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
* API deployment (Flask / FastAPI)
* Model versioning
* CI/CD pipeline
* Visualization dashboard

---

## 📚 Dataset

California Housing Dataset with features like:

* Median Income
* Housing Median Age
* Total Rooms
* Population
* Ocean Proximity

---

## 🤝 Contribution

Contributions are welcome. Feel free to fork and improve the project.

---

* You pasted it inside a code block in GitHub
* Or file name is not exactly `README.md`

If you want, I can upgrade this into a **premium GitHub README (with badges, visuals, and recruiter appeal)**.
