# ================================================================================================================================================================================================================================================================
# Imports

import csv
import numpy as np
import pickle as pkl
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
        else:
            file = request.files['file']
        
        try:
            with file.stream as f:
                reader = csv.reader(f)
                new_header = next(reader)
                new_data = [row for row in reader]

            with open('./data/original_data.csv', 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                original_header = next(reader)
                original_data = [row for row in reader]

            if new_header != original_header:
                return jsonify({'Error': 'The new dataset is missing or has extra columns'}), 400
            
            updated_data = original_data + new_data
            updated_clean = utils.process_data(original_header, updated_data)

            feature_idxs = [i for i, col in enumerate(original_header) if col not in ['date','precipitation']]
            date_idx = original_header.index('date')
            precip_idx = original_header.index('precipitation')

            X = []
            y = []
            for row in updated_clean:
                feature_values = [float(row[i]) for i in feature_idxs]
                target_values = float(row[precip_idx])
                X.append(feature_values)
                y.append(target_values)

            X = np.array(X)
            y = np.array(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

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

                with open('./data/updated_data.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(original_header)
                    writer.writerows(updated_data)

                pkl.dump(model, open('./models/updated_model.pkl', 'wb'))

                return jsonify({'Message': 'Updated dataset and model saved successfully'}), 200

            except Exception as e:
                return jsonify({'Error': f'Error during Saving: {str(e)}'}), 500    
    
    except Exception as e:
        return jsonify({'Error': str(e)}), 500
    
    
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