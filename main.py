import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

#Joblib will save trained model and pipeline in these pkl files.
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attributes, cat_attributes):
    # Numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    # Full pipeline
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attributes),
        ('cat', cat_pipeline, cat_attributes)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Train the model
    housing = pd.read_csv('housing_data.csv')

    # Create Stratified test set based on income category
    housing['income_cat'] = pd.cut(
        housing['median_income'],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.loc[test_index].drop('income_cat', axis=1).to_csv('input.csv', index=False)
        housing = housing.loc[train_index].drop('income_cat', axis=1)

    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value', axis=1)

    num_attributes = housing_features.drop('ocean_proximity', axis=1).columns.tolist()
    cat_attributes = ['ocean_proximity']

    pipeline = build_pipeline(num_attributes, cat_attributes)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("model is trained!")

else:
    # lets do inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')
    transformed_data = pipeline.transform(input_data)
    predictions = model.predict(transformed_data)
    input_data['median_house_value'] = predictions

    input_data.to_csv('output.csv', index=False)
    print('Inference is completed, Results saved to output.csv')