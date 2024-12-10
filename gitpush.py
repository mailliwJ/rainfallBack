# Push Saved Files to GitHub

import base64
import os
import requests


# GitHub repository details
REPO_OWNER = 'mailliwj'
REPO_NAME = 'rainfallBack'
BRANCH = 'main'
FILE_PATH = 'models/best_model.pkl'  # Path in the repo where the file will be saved
LOCAL_FILE = "best_model.pkl"         # Local file to be uploaded

# Get GitHub token from environment variable
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def push_file_to_github(FILE, FILE_PATH):
    # GitHub API URL for the file
    API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    try:
        # Read the local file
        with open(FILE, "rb") as file:
            content = file.read()
        
        # Base64 encode the content
        encoded_content = base64.b64encode(content).decode("utf-8")
        
        # Check if the file already exists on GitHub
        response = requests.get(API_URL, headers={"Authorization": f"token {GITHUB_TOKEN}"})
        
        if response.status_code == 200:
            # If the file exists, get the SHA for updating
            sha = response.json()["sha"]
        else:
            # If the file doesn't exist, set SHA to None
            sha = None

        # Create the payload
        payload = {
            "message": "Upload retrained model", # Commit message
            "content": encoded_content,          # Base64-encoded content
            "branch": BRANCH                     # Branch to push to
        }
        if sha:
            payload["sha"] = sha  # Include SHA if the file already exists

        # Send the request to GitHub
        response = requests.put(API_URL, json=payload, headers={"Authorization": f"token {GITHUB_TOKEN}"})
        
        if response.status_code in [200, 201]:
            print("File uploaded successfully!")
        else:
            print(f"Failed to upload file: {response.json()}")

    except Exception as e:
        print(f"An error occurred: {e}")