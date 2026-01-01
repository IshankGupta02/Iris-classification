from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        try:
            # Get data from form
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
        except ValueError:
            return render_template('index.html', prediction_text='Error: Please enter valid numeric values for all fields.')

        # Create numpy array for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]

        # Handle prediction output
        if isinstance(prediction, str):
            result = prediction  # Directly use the string if the model returns class names
        else:
            # Map output (0, 1, 2) to class names
            classes = ['Setosa', 'Versicolor', 'Virginica']
            result = classes[int(prediction)]

        return render_template('index.html', prediction_text=f'Predicted Species: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)