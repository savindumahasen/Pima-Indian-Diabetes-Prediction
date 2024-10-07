from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your ML model (ensure the file path and format are correct)
model_path = "./pickle.h5"  # assuming it's a pickle model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define feature names in order
        feature_names = ['0', '1', '2', '3', '4', '5', '6', '7']

        # Extract features from form data
        features = [float(request.form.get(f)) for f in feature_names]
        input_data = [features]

        # Perform prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]  # Assuming the model has predict_proba

        # Set result based on prediction
        if prediction == 0:
            result = "Diabetes is not predicted"
        else:
            result = "Diabetes is predicted"

        return jsonify({'result': result, 'probability': probabilities[1]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
