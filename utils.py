# =====================================================================================================================================================================
# Imports
import json
import numpy as np
import pickle as pkl

from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

from time import time

# =====================================================================================================================================================================
# Handle Data

def process_data(header, data_rows):
    """
    Process raw CSV files into numeric features(X) and target(y) arrays.

    Steps:
    - Identifies precipitation column index
    - Filters out date and snow_depth columns
    - Removes rows with missing values
    - Converts all remaining values to floats
    - Separates precipitation (y) from the other features(X)

    Returns:
    X (np.ndarray): Features matrix
    y (np.ndarray): Target vector (in this case, precipitation)
    """
    # Get precipitation original index in header
    precip_col_idx = header.index('precipitation')

    # Get the column indices for the desired features in the dataset
    feature_idxs = [i for i, feat_name in enumerate(header) if feat_name not in ['date', 'snow_depth']]
    
    # Get precipitation index in filtered feature set
    precip_idx = feature_idxs.index(precip_col_idx)

    # Define missings/nan value markers
    missings = [None, '', 'NaN', 'nan', 'N/A', 'n/a', 'NULL', 'null']
    
    # Initialise list to add value lists (rows) to
    all_values = []

    # Iterate through each data row
    for row in data_rows:
        # Extract only desired columns
        values = [row[i] for i in feature_idxs]
        # Skip any rows with missings
        if any(val in missings for val in values):
            continue
        # Convert al values to floats
        values = [float(val) for val in values]
        # Add cleaned row to al_values
        all_values.append(values)

    # Initialise X, y lists
    X = []
    y = []
    
    # Iterate over rows and remove precipitation value from features
    for row in all_values:
        feature_vals = [val for i, val in enumerate(row) if i != precip_idx]
        target_val = row[precip_idx]
        X.append(feature_vals)
        y.append(target_val)

    # Convert lists to arrays for modeling
    X = np.array(X)
    y = np.array(y)

    return X, y

# =====================================================================================================================================================================
# Cross-Validate

def cross_validate_models(algorithm_list, X_train, y_train, scaler) -> dict:
    """
    Performs cross_validation on a dictionary of algorithms {name: algorithm}
    Uses the scaler used in initial training of first model
    Uses predefined metrics for cross-validation
    
    Steps:
    - For each algorithm in the dictionary, creates a pipeline with the scaler and the algorithm
    - Performs cross-validation and records MSE, RMSE and MAPE)
    
    Returns:
    cv_results (dict): Dictionary of model names and their mean error metrics)
    pipelines (dict): Dictionary of each models name and pipeline.
    """
    cv_results = {}
    pipelines = {}

    for alg in algorithm_list:
        # Create a pipeline for each algorithm
        pipe = Pipeline(steps=[
            ('scaler', scaler),
            ('regressor', alg)
        ])

        # Perform cross-validation
        cv_scores = cross_validate(pipe, X_train, y_train, scoring=('neg_mean_squared_error','neg_root_mean_squared_error'))

        # Record metrics as a nested dictionary
        cv_results[alg.__class__.__name__] = {
            'MSE': -np.mean(cv_scores['test_neg_mean_squared_error']),
            'RMSE': -np.mean(cv_scores['test_neg_root_mean_squared_error']),
        }

        pipelines[alg.__class__.__name__] = pipe

    return cv_results, pipelines

# =====================================================================================================================================================================
# Tune Hyperparams

def tune_hyperparameters(pipelines: dict, param_grids: dict, X_train, y_train, cv_scoring: str) -> dict:
    """
    Performs hyperparameter tuning on pipelines using GridSearchCV.
    Allows user defined coring metric
    
    Steps:
    - For each pipeline, checks if there's a parameter grid
    - Run a grid search to find the best hyperparameters
    - If no grid is found, just fits the pipeline directly
    - Prints timing and best parameters for each model
    
    Returns:
    tuned_models (dict): Dictionary of model names and tuned models
    """
    tuned_models = {}

    for name, pipe in pipelines.items():

        params = param_grids.get(name)
        if params:
            print(f'Tuning {name} hyperparameters...')
            gs = GridSearchCV(pipe, param_grid=params, cv=5, scoring=cv_scoring)

            start = time()
            gs.fit(X_train, y_train)
            end = time()

            tuning_time = end - start
            time_message = (f'Tuning {name} took: {tuning_time:.3f} seconds' if tuning_time < 60 else f'Tuning {name} took: {tuning_time/60:.3f} minutes')

            best = gs.best_estimator_

            print(f'----Hyperparameter tuning complete ----')
            print(time_message)
            if 'neg_' in cv_scoring:
                print(f'Best Score: {-gs.best_score_:.5f}')
            else:
                print(f'Best Score: {gs.best_score_:.5f}')
            print(f'Best parameters:\n{gs.best_params_}')
            print()
        
        else:
            print(f'No parameter grid found for {name}. Fitting model directly...')
            
            start = time()
            cv = cross_validate(pipe, X_train, y_train, scoring=cv_scoring)
            pipe.fit(X_train, y_train)
            best = pipe
            end = time()

            tuning_time = end - start
            time_message = (f'Fitting {name} took: {tuning_time:.3f} seconds' if tuning_time < 60 else f'Fitting {name} took: {tuning_time/60:.3f} minutes')
            print(time_message)


            # Prints best scores from cross-validation
            if 'neg_' in cv_scoring:
                print(f"Best Score: {-np.mean(cv['test_score']):.5f}")
            else:
                print(f"Best Score: {np.mean(cv['test_score']):.5f}")
            print()
        
        tuned_models[name] = best

    return tuned_models

# =====================================================================================================================================================================
# Evaluate on test set

def test_evaluation(tuned_models: dict, X_train=None, y_train=None, X_test=None, y_test=None) -> dict:
    """
    Fits the tuned models on training data and evaluates them on the test set.
    
    Steps:
    - If tuned_models is a single model (pipeline), convert it to a dict.
    - Fit each model on X_train, y_train.
    - Predict on X_test and compute MSE, RMSE, MAPE.

    Returns:
    - evaluation_results (dict): Dictionary of evaluation results for each model.
    """
    # Accounts for when i want to use this function and there is only one model
    # This will actually be the majority use case but i wanted to make a function that allows for evaluation of mutiple models
    if not isinstance(tuned_models, dict):
        tuned_models = {f'{tuned_models.steps[-1][1].__class__.__name__}': tuned_models}

    evaluation_results = {}

    for name, model in tuned_models.items():
        # Fit on training data
        model.fit(X_train, y_train)
        # Predict on test data
        y_preds = model.predict(X_test)

        # Calculate evalution metrics
        mse_score = mean_squared_error(y_test, y_preds)
        rmse_score = root_mean_squared_error(y_test, y_preds)

        # Organise into dictionary and return metrics
        evaluation_results[name] = {
            'MSE': mse_score,
            'RMSE': rmse_score
            }

    return evaluation_results

# =====================================================================================================================================================================
# Save best model

def save_best_model(evaluation_results: dict, tuned_models, selection_metric=''):
    """
    Selects the best model based on a specified metric and saves it
    
    Steps:
    - If tuned_models is a dictionary, finds the model with the best (lowest) metric score
    - If it's a single model, just uses that model
    - Save the selected model as a pickle file

    Returns:
    best_model (pipeline): Pipeline of best performing estimator
    """
    if not isinstance(tuned_models, dict):
        best_model = tuned_models

    else:
        # Finds the index of the best model based on the specified selection_metric
        # https://tinyurl.com/minfunction
        best_model_name = min(evaluation_results, key=lambda name: evaluation_results[name][selection_metric])
        best_model = tuned_models[best_model_name]

    # Saves the best model
    pkl.dump(best_model, open(f'./models/model.pkl', 'wb'))

    return best_model

# =====================================================================================================================================================================
# Save evaluation_results

def save_evaluation_results(model, results):
    """
    Saves evaluation results (a dictionary) to a JSON file.
    """
    model_name = model.steps[-1][1].__class__.__name__
    if model_name in results:
        model_evaluation = {
            'Model': model_name,
            'MSE': results[model_name].get('MSE'),
            'RMSE': results[model_name].get('RMSE'),
            }
        
    json.dump(model_evaluation, open('./data/evaluation_results.json', 'w'), indent=4)

    return model_evaluation

# =====================================================================================================================================================================
# Load evaluation_results

def load_evaluation_results():
    """
    Loads previously saved evaluation results from a JSON file.
    If the file doesn’t exist, returns None.
    """
    try:
        with open('./data/evaluation_results.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return None
    
# =====================================================================================================================================================================
