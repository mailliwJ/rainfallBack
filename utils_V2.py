import json
import numpy as np
import os
import pickle as pkl

from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

from time import time

from zipfile import ZipFile

# =====================================================================================================================================================================
# Handle Data

def process_data(header, data_rows):
    pressure_idx = header.index('pressure')
    cleaned_data = []
    missings = [None, '', 'NaN', 'nan', 'N/A', 'n/a', 'NULL', 'null']

    for row in data_rows:
        if any(val in missings for val in row):
            continue

        try:
            pressure_values = float(row[pressure_idx]) / 1000
        except ValueError:
            continue

        row[pressure_idx] = pressure_values

        cleaned_data.append(row)

    return cleaned_data

# =====================================================================================================================================================================
# Cross-Validate

scaler = pkl.load(open('./transformers/scaler.pkl', 'rb'))

def cross_validate_models(models: dict, X_train, y_train) -> dict:

    model_names = []
    mse = []
    rmse = []
    mape = []
    pipelines = {}

    for name, alg in models.items():

        pipe = Pipeline(steps=[
            ('scaler', scaler),
            ('regressor', alg)
        ])

        cv_scores = cross_validate(pipe, X_train, y_train, scoring=('neg_mean_squared_error','neg_root_mean_squared_error','neg_mean_absolute_percentage_error'))
        
        model_names.append(name)
        mse.append(-np.mean(cv_scores['test_neg_mean_squared_error']))
        rmse.append(-np.mean(cv_scores['test_neg_root_mean_squared_error']))
        mape.append(-np.mean(cv_scores['test_neg_mean_absolute_percentage_error']))
        pipelines[name] = pipe

    cv_results = {
        'Model': model_names,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

    return cv_results, pipelines

# =====================================================================================================================================================================
# Tune Hyperparams

def tune_hyperparameters(pipelines: dict, param_grids: dict, X_train, y_train, cv_scoring: str) -> dict:
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

            if 'neg_' in cv_scoring:
                print(f'Best Score: {-np.mean(cv['test_score']):.5f}')
            else:
                print(f'Best Score: {np.mean(cv['test_score']):.5f}')
            print()
        
        tuned_models[name] = best

    return tuned_models

# =====================================================================================================================================================================
# Evaluate on test set

def test_evaluation(tuned_models: dict, X_train=None, y_train=None, X_test=None, y_test=None) -> dict:
    model_names = []
    mse = []
    rmse = []
    mape = []

    if not isinstance(tuned_models, dict):
        tuned_models = {f'{tuned_models.steps[-1][1].__class__.__name__}': tuned_models}

    for name, model in tuned_models.items():

        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)

        mse_score = mean_squared_error(y_test, y_preds)
        rmse_score = root_mean_squared_error(y_test, y_preds)
        mape_score = mean_absolute_percentage_error(y_test, y_preds)

        model_names.append(name)
        mse.append(mse_score)
        rmse.append(rmse_score)
        mape.append(mape_score)

    evaluation_results = {
        'Model': model_names,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

    return evaluation_results

# =====================================================================================================================================================================
# Save best model

def save_best_model(evaluation_results: dict, tuned_models: dict, selection_metric='', save_path='./models/best_model.pkl'):
    
    if not isinstance(tuned_models, dict):
        best_model = tuned_models
    else:
        min_idx = np.argmin(evaluation_results[selection_metric])
        best_model_name = evaluation_results['Model'][min_idx]
        best_model = tuned_models[best_model_name]

    pkl.dump(best_model, open(save_path, 'wb'))

    return best_model

# =====================================================================================================================================================================
# Load evaluation_results

def load_evaluation_results():
    try:
        with open('./data/evaluation_results.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return None
    
# =====================================================================================================================================================================
# Save evaluation_results

def save_evaluation_results(results):

    with open('./data/evaluation_results.json', 'w') as file:
        json.dump(results, file, indent=4)

# =====================================================================================================================================================================


