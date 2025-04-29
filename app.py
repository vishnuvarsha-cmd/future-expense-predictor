from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
# Load the trained model (ensure 'model.pkl' is in the same directory)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read inputs from form
    year = int(request.form['year'])
    month = int(request.form['month'])
    day_of_week = int(request.form['day_of_week'])
    days = int(request.form['days'])
    
    # Prepare the feature array for prediction
    features = np.array([[year, month, day_of_week, days]])
    
    # Predict log-transformed expense and convert back to actual expense
    log_pred = model.predict(features)
    prediction = np.exp(log_pred)[0]  # Convert from log scale
    
    # Format prediction to 2 decimal places
    output = round(prediction, 2)
    
    # Render the result on the webpage
    return render_template('index.html', prediction_text=f'Predicted Expense: ${output}')

if __name__ == '__main__':
    app.run(debug=True)
