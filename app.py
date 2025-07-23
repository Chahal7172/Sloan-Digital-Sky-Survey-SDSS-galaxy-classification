from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- LOAD THE SAVED FILES ---
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
label_classes = joblib.load('label_classes.pkl')

# --- DEFINE THE ROUTES ---
@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, makes a prediction, and renders the result page."""
    
    # Get the feature values from the form, convert to float
    # The order MUST match the features used during training: u, g, r, i, z, redshift
    features = [float(x) for x in request.form.values()]
    
    # Convert to a 2D numpy array
    final_features = np.array(features).reshape(1, -1)
    
    # Scale the input features
    scaled_features = scaler.transform(final_features)
    
    # Make a prediction
    prediction_index = model.predict(scaled_features)[0]
    
    # Get the corresponding class name
    prediction_class = label_classes[prediction_index]
    
    # Render the output page
    return render_template('output.html', prediction_text=f'The object is classified as: {prediction_class}')

# --- RUN THE APPLICATION ---
if __name__ == "__main__":
    app.run(debug=True)
