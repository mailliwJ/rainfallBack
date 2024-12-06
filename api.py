# ================================================================================================================================================================================================================================================================
# Imports

import numpy as np
import pickle as pkl
import pandas as pd
import subprocess
import os
import utils as utils

from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split


# ================================================================================================================================================================================================================================================================

app = Flask(__name__)

model = pkl.load(open('./models/best_model.pkl', 'rb'))
scaler = pkl.load(open('./transformers/scaler.pkl', 'rb'))

# ================================================================================================================================================================================================================================================================

@app.route('/')
def home():
    return 'Welcome to my rain prediction app'

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    required_params = ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'pressure', 'snow_depth']
    if not all(param in data for param in required_params):
        return jsonify({'Error': 'Missing required parameter. Must provide a value for all parameters'})
    
    try:
        cloud_cover = float(data['cloud_cover'])
        sunshine = float(data['sunshine'])
        global_radiation = float(data['global_radiation'])
        max_temp = float(data['max_temp'])
        mean_temp = float(data['mean_temp'])
        min_temp = float(data['min_temp'])
        pressure = float(data['pressure'])
        snow_depth = float(data['snow_depth'])


        input_data = np.array([[cloud_cover, sunshine, global_radiation, max_temp, mean_temp, min_temp, pressure, snow_depth]])
        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)[0]

        return jsonify({'Prediction': prediction})
    
    except Exception as e:
        return jsonify({'Error': str(e)}), 500
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/retrain_save', methods=['POST', 'PUT'])
def retrain():
    try:
        if 'file' not in request.files:
            return jsonify({'Error': 'No file uploaded'}), 400
        
        try:
        
            new_data = pd.read_csv(request.files['file'])
            original_data = pd.read_csv('./data/original_data.csv')

            if not all(col in original_data.columns for col in new_data.columns):
                return jsonify({'Error': 'The new dataset is missing or has extra columns'})

            updated_data = pd.concat([original_data, new_data], ignore_index=True)
            updated_clean = utils.process_data(updated_data)

            X_train, X_test, y_train, y_test = train_test_split(updated_clean.drop(['date', 'precipitation'], axis=1), updated_clean['precipitation'], test_size=0.2, random_state=13)

        except Exception as e:
            return jsonify({'Error': f'Error during preprocessing: {str(e)}'}), 500
        
        if request.method == 'POST':
            try:
                current_metrics = utils.load_evaluation_results()
                new_metrics = utils.test_evaluation(model, X_train, y_train, X_test, y_test)

                return jsonify({
                    'Message': 'Evaluation Complete. Metrics comparison ready',
                    'Current Evaluation Metrics': current_metrics,
                    'New Evaluation Metrics': new_metrics
                }), 200
        
            except Exception as e:
                return jsonify({'Error': f'Error during evaluation: {str(e)}'}), 500
            
        elif request.method == 'PUT':
            try:
                model.fit(X_train, y_train)
                updated_data.to_csv('./data/updated_data.csv', index=False)
                pkl.dump(model, open('./models/updated_model.pkl', 'wb'))

                return jsonify({'Message': 'Updated dataset and model saved successfully'})

            except Exception as e:
                return jsonify({'Error': f'Error during Saving: {str(e)}'}), 500    
    
    except Exception as e:
        return jsonify({'Error': str(e)}), 500
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/save', methods=['POST'])
def save():
    new_data = pd.read_csv(request.files['file'])
    original_data = pd.read_csv('./data/original_data.csv')

    updated_data = pd.concat([original_data, new_data], ignore_index=True)
    updated_clean = utils.process_data(updated_data)

    X_train, y_train = updated_clean.drop(['date', 'precipitation'], axis=1), updated_clean['precipitation']
    model.fit(X_train, y_train)

    updated_data.to_csv('./data/updated_data.csv', index=False)
    pkl.dump(model, open('./models/best_model.pkl', 'wb'))

    return jsonify({'Message': 'Dataset and model saved successfully'})
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
# ================================================================================================================================================================================================================================================================

# WebHook

# Path to repositorio and WSGI configuration
REPO_PATH = '/home/mailliwj/rainfall'
SERVER_PATH = '/var/www/mailliwj_pythonanywhere_com_wsgi.py' 

@app.route('/webhook', methods=['POST'])
def webhook():

    # Check request contains json data
    if not request.is_json:
        return jsonify({'Message': 'The request does not contain valid JSON data'}), 400
    
    payload = request.json
        # Check payload has repo information
    if 'repository' not in payload:
        return jsonify({'Message': 'No repository information found in the payload'}), 400
    
    repo_name = payload['repository']['name']
    # CLONE_URL = payload['repository']['clone_url']
        
    # Try to change to repo directory
    try:
        os.chdir(REPO_PATH)
    except FileNotFoundError:
        return jsonify({'Message': f'The repo directory {REPO_PATH} does not exist'}), 404

    # Perform git pull. Might need to add , CLONE_URL back into subprocess 1
    try:
        subprocess.run(['git', 'pull'], check=True)
        subprocess.run(['touch', SERVER_PATH], check=True) # Reload PythonAnywhere WebServer
        return jsonify({'Message': f'Successfully pulled updates from the repository {repo_name}'}), 200
    
    except subprocess.CalledProcessError as e:
        return jsonify({'Message': f'Error during git pull: {str(e)}'}), 500

# ===================================================================================================================================================================


if __name__ == '__main__':
    app.run(debug=True)