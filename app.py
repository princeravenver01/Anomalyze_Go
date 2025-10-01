from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from utils.preprocessing import load_and_preprocess_data
import joblib
import io

app = Flask(__name__)

#Configure models folder
MODELS_FOLDER = 'models'

#Global variables for the model and data features
kmeans_model = None
scaler = None
data_columns = None
training_distances = None # To hold distances from the original training data

#Paths for saved model files
MODEL_PATH = os.path.join(MODELS_FOLDER, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')


def load_model():
    """Loads the pre-trained model and associated files from disk."""
    global kmeans_model, scaler, data_columns
    
    #In a serverless environment, these files MUST exist as part of the deployment.
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH]):
        #This error will be shown in Vercel logs if files are missing
        raise FileNotFoundError("Model files not found. Please ensure the model is trained and files are in the 'models' directory before deploying.")

    print("Loading saved model from disk...")
    kmeans_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    data_columns = joblib.load(COLUMNS_PATH)
    print("Model loaded successfully.")


@app.route('/')
def index():
    return render_template('index.html', results=None)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        #Process file in memory instead of saving to disk
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        
        #Keep original data for display
        stream.seek(0)
        df_test_original = pd.read_csv(stream, header=None)
        
        #Reset stream position to read again for preprocessing
        stream.seek(0)
        df_test = load_and_preprocess_data(stream)

        #Ensure the test data has the same columns as the training data
        df_test = df_test.reindex(columns=data_columns, fill_value=0)

        #Scale the test data
        df_test_scaled = scaler.transform(df_test)

        #Calculate distances to the nearest cluster
        distances = kmeans_model.transform(df_test_scaled).min(axis=1)

        #Calculate threshold from the training data's normal instances
        #We need to re-calculate the distances on the training data to set the threshold
        train_data_path = 'data/KDDTrain+.txt'
        df_train = load_and_preprocess_data(train_data_path)
        df_normal = df_train[df_train['label'] == 'normal'].drop('label', axis=1)
        df_normal_scaled = scaler.transform(df_normal)
        training_distances = kmeans_model.transform(df_normal_scaled).min(axis=1)
        threshold = np.percentile(training_distances, 99)

        #Identify anomalies
        anomalies_mask = distances > threshold
        anomalies = df_test_original[anomalies_mask]

        return render_template('index.html', results=anomalies)

    return "File upload failed"

#Load the model when the application starts
load_model()