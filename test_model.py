import pandas as pd
import pytest
from ml.data import process_data
from ml.model import load_artifact 
from config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

@pytest.fixture
def data():
    """Load the data to feed in the test cases.
    Returns:
        df: DataFrame
    """
    df = pd.read_csv(f'{DATASET_DIR}/{config.app_config.data_file}')
    df.columns = [i.strip() for i in df.columns]
    return df

@pytest.fixture
def artifact_encoder():
    """Load the encoder to feed in the test cases.

    Returns:
        encoder: OneHotEncoder
    """
    encoder = load_artifact(f'{TRAINED_MODEL_DIR}/{config.app_config.encoder_file_name}')
    return encoder

@pytest.fixture
def artifact_lb():
    """Load the encoder to feed in the test cases.
    Returns:
        lb: LabelBinarizer
    """
    lb = load_artifact(f'{TRAINED_MODEL_DIR}/{config.app_config.binarizer_file_name}')
    return lb

def test_process_data_train(data):
    """ Test the shape the data for training.
    """
    X, _, _, _ = process_data(
        data, categorical_features=config.model_config.cat_features,
        label=config.model_config.target, training=True
    )
    assert X.shape[0] == data.shape[0]

def test_process_data_test(data, artifact_encoder, artifact_lb):
    """ Test the shape of the data for testing.
    """
    X, _, _, _ = process_data(
        data, categorical_features=config.model_config.cat_features,
        label=config.model_config.target, training=False,
        encoder = artifact_encoder, lb = artifact_lb
    )
    assert X.shape[0] == data.shape[0]

def test_process_data_target(data, artifact_encoder, artifact_lb):
    """ Test the shape of the target
    """
    _, y, _, _ = process_data(
        data, categorical_features=config.model_config.cat_features,
        label=config.model_config.target, training=False,
        encoder = artifact_encoder, lb = artifact_lb
    )
    assert y.shape[0] == data.shape[0]

def test_load_artifact_lb_type():
    """ Test the type of the encoder
    """
    lb = load_artifact(f'{TRAINED_MODEL_DIR}/{config.app_config.binarizer_file_name}')
    assert type(lb).__name__ == "LabelBinarizer"

def test_load_artifact_encoder_type():
    """ Test the type of the encoder
    """
    encoder = load_artifact(f'{TRAINED_MODEL_DIR}/{config.app_config.encoder_file_name}')
    assert type(encoder).__name__ == "OneHotEncoder"