# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import sys
# Add the necessary imports for the starter code.
from ml.model import train_model, compute_model_metrics, inference, save_artifact
from ml.data import process_data
from config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

# Add code to load in the data.
data = pd.read_csv(f'{DATASET_DIR}/{config.app_config.data_file}')
data.columns = [i.strip() for i in data.columns]
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=config.model_config.test_size,
                               random_state = config.model_config.random_state)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=config.model_config.cat_features,
    label=config.model_config.target, training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=config.model_config.cat_features,
    label=config.model_config.target, training=False,
    encoder = encoder, lb = lb
)

# Train and save a model.
model_trained = train_model(X_train, y_train)
save_artifact(model_trained, f'{TRAINED_MODEL_DIR}/{config.app_config.model_file_name}')
save_artifact(encoder, f'{TRAINED_MODEL_DIR}/{config.app_config.encoder_file_name}')
save_artifact(lb, f'{TRAINED_MODEL_DIR}/{config.app_config.binarizer_file_name}')

preds_train = inference(model_trained, X_train)
precision_train, recall_train, fbeta_train = compute_model_metrics(y_train, preds_train)
print(recall_train)
print(precision_train)

preds_test = inference(model_trained, X_test)
precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, preds_test)
print(recall_test)
print(precision_test)
