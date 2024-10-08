import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report  # Import model evaluation metrics
import os

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("C:\\Users\\Raman\\SEM V project\\mini-project\\CPS\\model.pkl\\Crop_data.csv")


# Separate features (X) and labels (y)
X = data.drop(columns=['label'])
y = data['label']

# Create and train a Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        n = float(request.form['n'])
        p = float(request.form['p'])
        k = float(request.form['k'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create a data frame from user input
        user_input = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]],
                                    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        print("User Input Data:", user_input)

        # Make a prediction using the trained model
        prediction = dt_model.predict(user_input)

        # Model Evaluation (Add this code here)
        y_pred_train = dt_model.predict(X)

        # Evaluate model performance on training data
        accuracy_train = accuracy_score(y, y_pred_train)
        report_train = classification_report(y, y_pred_train)

        print(f"Training Accuracy: {accuracy_train}")
        print("Classification Report (Training):\n", report_train)

# Return the prediction result as JSON response
        return jsonify({'prediction': prediction[0]})
if __name__ == '__main__':
    app.run(debug=True)
