# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
#filename =
classifier = pickle.load(open('random_forest_classifier_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        residual_sugar= float(request.form['residual_sugar'])
        chlorides= float(request.form['chlorides'])
        free_sulfur_dioxide= int(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide= int(request.form['total_sulfur_dioxide'])
        density= float(request.form['density'])
        pH=float(request.form['pH'])
        sulphates= float(request.form['sulphates'])
        alcohol= float(request.form['alcohol'])

        
        data = np.array([[fixed_acidity, volatile_acidity,citric_acid,residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,pH,sulphates,alcohol]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)