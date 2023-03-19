from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from joblib import dump, load
from ml.data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    space = dict()
    space['n_estimators'] = [100, 200, 300, 400, 500]
    space['max_depth'] = [3, 5, 7, 10]
    space["class_weight"] = ["balanced"]
    
    cv_repeat = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=1)
    model = RandomForestClassifier()
    search = RandomizedSearchCV(
        model, space, n_iter=20, scoring='f1',
        cv=cv_repeat, random_state=94,
        refit = True)
    result = search.fit(X_train, y_train)
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    # model.set_params(**result.best_params_)
    return result.best_estimator_

def save_artifact(artifact, path):
    """

    Saves the artifact into a define storage.
    Inputs
    ------
    artifact : Scikit-learn object
    Returns
    -------
    None    
    """
    dump(artifact, path)

def load_artifact(path):
    """
    Load the model into the application.
    Inputs
    ------
    model : RandomForest
    Returns
    -------
    Scikit Learn Model    
    """
    model = load(path)
    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def model_slice(model, X, feature, cat_features, encoder, lb):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    feature : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    y : np.array
        Target used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    dict_slices = {}
    for held_category in X[feature].unique():
        X_slice = X.loc[X[feature] == held_category, :]
        X_test, y_test, _, _ = process_data(
            X_slice, categorical_features=cat_features, label="salary", training=False,
            encoder = encoder, lb = lb
        )
        preds = model.predict(X_test)
        precision_slice, recall_slice, fbeta_slice = compute_model_metrics(y_test, preds)
        dict_slices[held_category] = {'precision':precision_slice,
                                      'recall':recall_slice,
                                      'fbeta':fbeta_slice}
    return dict_slices