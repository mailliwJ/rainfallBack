# ================================================================================================================================================================================================================================================================
# Imports

import csv
import io
import numpy as np
import pickle as pkl
import subprocess
import os
import utils_V2 as utils

from flask import Flask, jsonify, redirect, request, url_for
from sklearn.model_selection import train_test_split

# ================================================================================================================================================================================================================================================================

app = Flask(__name__)

# ================================================================================================================================================================================================================================================================
# Check
# Basic home route just to return a welcome message for now
@app.route('/home')
def home():
    return 'Welcome to RainFall'

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/')
def base():
    return redirect(url_for('home'))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Endpoint takes in climatic parameters via a JSON POST request and returns a rainfall prediction in mm

@app.route('/predict', methods=['POST'])
def predict():
    
    # Load trained model and scaler that was fitted and used to transform training data
    model = pkl.load(open('./models/model.pkl', 'rb'))
    scaler = pkl.load(open('./transformers/scaler.pkl', 'rb'))

    # Parse JSON data from request
    data = request.get_json()

    # Define required parameters for the prediction
    required_params = ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'pressure']
    
    # Check all required parameters are in the incoming JSON
    if not all(param in data for param in required_params):
        return jsonify({'Error': 'Missing required parameter. Must provide a value for all parameters'})
    
    try:
        # Convert input paramters to floats
        cloud_cover = float(data['cloud_cover'])
        sunshine = float(data['sunshine'])
        global_radiation = float(data['global_radiation'])
        max_temp = float(data['max_temp'])
        mean_temp = float(data['mean_temp'])
        min_temp = float(data['min_temp'])
        pressure = float(data['pressure']*1000)

        # Make a single input data array for the model to work with
        input_data = np.array([[cloud_cover, sunshine, global_radiation, max_temp, mean_temp, min_temp, pressure]])
        
        # Scale the input data using the scaler that was fitted when training the model
        scaled_input = scaler.transform(input_data)

        # Make a prediction using the trained model. As .predict() returns an array, need to add [0] to access the prediction
        prediction = model.predict(scaled_input)[0]
        
        # Return the prediction in a JSON. This is the JSON that is returned back to the frontend
        return jsonify({'Prediction': prediction})
    
    # If an error occurs somewhere in this process returns a 500 error with the specific exception
    except Exception as e:
        return jsonify({'Error': str(e)}), 500
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# This endpoint handles retraining a model with new data and the saving of new models and updated data.
# - action: 'evaluate': Evaluates the model, retrained on updated data and returns the evaluation metrics
# - action: 'save': Saves the updated data and retrained model

@app.route('/retrain_save', methods=['POST', 'GET'])
def retrain():
    
    model = pkl.load(open('./models/model.pkl', 'rb'))

    try:
        # Checks if the request actually contains a file (aiming to send a csv file)
        if 'file' not in request.files:
            return jsonify({'Error': 'No file uploaded'}), 400
        else:
            file = request.files['file']
        
        try:
            # Get CSV content using io.StringIO
            file_content = file.read().decode('utf-8')
            f = io.StringIO(file_content)

            # Read the new CSV file
            reader = csv.reader(f)
            new_header = next(reader)   # Extract the header
            new_data = [row for row in reader]  # Extract the data
            print('New data read successfully')

            # Read the original and saved CSV file
            """with open('./data/new_data.csv', 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                new_header = next(reader)  # Extract the header
                new_data = [row for row in reader] # Extract the data
                print('Original data read successfully')"""

            # Read the original and saved CSV file
            with open('./data/original_data.csv', 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                original_header = next(reader)  # Extract the header
                original_data = [row for row in reader] # Extract the data
                print('Original data read successfully')

            # Check that the headers match to ensure data can be merged
            if new_header != original_header:
                return jsonify({'Error': 'The new dataset is missing or has extra columns'}), 400
            
            # Merge original and new data
            updated_data = original_data + new_data

            # Process the data using utils function process_data
            X, y = utils.process_data(original_header, updated_data)

            # Split data into X, y pairs for train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
            print('Data ready for retraining.')

        # If an error occurs somewhere in this process returns a 500 error with the specific exception
        except Exception as e:
            return jsonify({'Error': f'Error during preprocessing: {str(e)}'}), 500
        
        # Use a parameter to select what to action to take
        action = request.args.get('action', 'evaluate')

        if action == 'evaluate':
            try:
                current_metrics = utils.load_evaluation_results()   # Loads current metrics
                new_metrics = utils.test_evaluation(model, X_train, y_train, X_test, y_test)   # Calculates updated dataset evaluation metrics
            
                # Returns both sets of metrics to frontend for user evaluation and decision
                return jsonify({
                    'Message': 'Evaluation Complete. Metrics comparison ready',
                    'Current Evaluation Metrics': current_metrics,
                    'New Evaluation Metrics': new_metrics
                }), 200
        
            except Exception as e:
                return jsonify({'Error': f'Error during evaluation: {str(e)}'}), 500
        
        # If the request is PUT the model is fit to the new dataset again and both are saved permanently
        elif action == 'save':
            try:
                # Retrain the model on new combined dataset
                updated_model = model
                updated_model.fit(X_train, y_train)

                # Save new dataset
                with open('./data/updated_data.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(original_header)
                    writer.writerows(updated_data)  # Saving uncleaned but updated

                # Save the retrained model
                pkl.dump(updated_model, open('./models/model.pkl', 'wb'))

                utils.save_evaluation_results(new_metrics) 

                return jsonify({'Message': 'Updated dataset and model saved successfully'}), 200

            except Exception as e:
                return jsonify({'Error': f'Error during Saving: {str(e)}'}), 500    
    
    # Idea to put the logic for this whole block in a try except was to be able to catch any other issues i hadnt thought of
    except Exception as e:
        return jsonify({'Error': str(e)}), 500

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ================================================================================================================================================================================================================================================================

# WebHook

# Path to repositorio and WSGI configuration
REPO_PATH = '/home/mailliwj/rainfallBack'
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
    

    REPO_NAME = payload['repository']['name']
    CLONE_URL = payload['repository']['clone_url']
        
    # Try to change to repo directory
    try:
        os.chdir(REPO_PATH)
    except FileNotFoundError:
        return jsonify({'Message': f'The repo directory {REPO_PATH} does not exist'}), 404

    try:
        subprocess.run(['git', 'pull', CLONE_URL], check=True)  # Run git pull
        subprocess.run(['touch', SERVER_PATH], check=True)  # Reload PythonAnywhere WebServer
        return jsonify({'Message': f'Successfully pulled updates from the repository {REPO_NAME}'}), 200
    
    except subprocess.CalledProcessError as e:
        return jsonify({'Message': f'Error during git pull: {str(e)}'}), 500

# ===================================================================================================================================================================

# Some PythonAnywhere troubleshooting says not to include the app.run() call
if __name__ == '__main__':
    app.run(debug=True)