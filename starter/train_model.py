# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import sys
# Add the necessary imports for the starter code.
from ml.model import train_model, compute_model_metrics, inference, save_model
from ml.data import process_data

# Add code to load in the data.
data = pd.read_csv('../data/census.csv')
data.columns = [i.strip() for i in data.columns]
print(data.columns)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb = lb
)

# Train and save a model.
model_trained = train_model(X_train, y_train)
save_model(model_trained, '../model/model.pkl')

preds_train = inference(model_trained, X_train)
precision_train, recall_train, fbeta_train = compute_model_metrics(y_train, preds_train)


preds_test = inference(model_trained, X_test)
precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, preds_test)

