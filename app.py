# Import necessary libraries
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load data
heartdata = pd.read_csv('heart_disease_data.csv')


# Prepare data
x = heartdata.drop(columns=['target'], axis=1)  # Features (Input data)
y = heartdata['target']  # Target (Output data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train model
model = LogisticRegression()
model.fit(x_train, y_train)

# Save model
joblib.dump(model, 'Heart_Disease_Predictor.joblib')

# Initialize Flask app
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    input_data = request.form.to_dict()
    # Convert input data to numpy array
    input_data_as_numpy_array = np.array(list(input_data.values())).astype(float)
    # Reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    # Make prediction
    prediction = model.predict(input_data_reshaped)
    # Return prediction result
    if prediction[0] == 0:
        result = 'The Person does not have a Heart Disease'
    else:
        result = 'The Person has Heart Disease'
    return render_template('result.html', prediction=result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
