"""
    Module for data slicing and testing performance.
"""
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

from ml.model import model_slice, load_artifact, inference
from ml.data import process_data
from  pprint import pprint
import json
from config.core import DATASET_DIR,TRAINED_MODEL_DIR, config

data = pd.read_csv(f'{DATASET_DIR}/{config.app_config.data_file}')
data.columns = [i.strip() for i in data.columns]

model = load_artifact(f'{TRAINED_MODEL_DIR}/{config.app_config.model_file_name}')
encoder = load_artifact(f'{TRAINED_MODEL_DIR}/{config.app_config.encoder_file_name}')
lb = load_artifact(f'{TRAINED_MODEL_DIR}/{config.app_config.binarizer_file_name}')
print(type(lb))
print(type(lb).__name__)
sys.exit(0)
# Process the test data with the process_data function.
with open(f'{DATASET_DIR}/output/{config.app_config.output_file_slice}', 'w') as filee_txt:
    for feature in config.model_config.cat_features:
        dict_result = model_slice(model, data, feature, config.model_config.cat_features, encoder, lb)
        pprint(dict_result)
        filee_txt.write(f"{feature}")
        filee_txt.write("\n")
        filee_txt.write(json.dumps(dict_result))
        filee_txt.write("\n")