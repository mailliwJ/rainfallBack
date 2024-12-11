import csv
import numpy as np
import pickle as pkl
import utils_V2 as utils

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


with open('./data/original_data.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    original_header = next(reader)
    original_data = [row for row in reader]


X, y = utils.process_data(original_header, original_data)
print('X Shape:', X.shape)
print('y Shape:', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

algs_list = [
    LinearRegression(),
    Ridge(alpha=1),
    Lasso(alpha=5000),
    # ElasticNet(alpha=1.0, l1_ratio=0.5),
    # RandomForestRegressor(random_state=13, n_estimators=500),
    # MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', learning_rate='adaptive', max_iter=500, random_state=13)
    ]

scaler = StandardScaler()
cv_results, my_pipelines = utils.cross_validate_models(algs_list, X_train, y_train, scaler)

gs_params = {
    'Ridge': {
        'regressor__alpha': [0.01, 0.1, 1, 10, 100],
        'regressor__fit_intercept': [True, False],
        'regressor__solver': ['auto', 'svd', 'saga']
        },
    'Lasso': {
        'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'regressor__fit_intercept': [True, False],
        'regressor__max_iter': [1000, 5000, 10000, 20000]
        },
    'Elastic Net': {
        'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
        'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1]
        },
    'Random Forest': {
        'regressor__n_estimators': [50, 250, 500],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__bootstrap': [True, False]
        },
    'Mulit-Layer Perceptron': {
        'regressor__hidden_layer_sizes': [(64, 32), (128, 64, 32)],
        'regressor__activation': ['relu', 'tanh'],
        'regressor__solver': ['adam', 'sgd'],
        'regressor__alpha': [0.0001, 0.001],
        'regressor__learning_rate': ['constant', 'adaptive']
        }
    }

tuned_models = utils.tune_hyperparameters(my_pipelines, gs_params, X_train, y_train, 'neg_mean_squared_error')

evaluation_results = utils.test_evaluation(tuned_models, X_train, y_train, X_test, y_test)

best_model = utils.save_best_model(evaluation_results, tuned_models, selection_metric='MSE')

utils.save_evaluation_results(best_model, evaluation_results) 

pkl.dump(scaler, open('./transformers/scaler.pkl', 'wb'))

print('Model and transformers ready for use')