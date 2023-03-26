from flask import Flask, render_template, request
import requests
import pickle
import xgboost as xgb
import pandas as pd

app = Flask('app')

with open('my_model.pickle', 'rb') as file:
  xgb_model = pickle.load(file)


@app.route('/')
def home():
  return render_template('home_2.html')


@app.route('/predictions', methods=['POST'])
def predictions():
  input1 = float(request.form['Variance'])
  input2 = float(request.form['Skewness'])
  input3 = float(request.form['Curtosis'])
  input4 = float(request.form['Entropy'])

  # Convert the user inputs to a format that the XGBoost model expects
  input_array = [[input1, input2, input3, input4]]

  input_df = pd.DataFrame(
    input_array, columns=['Image.Var', 'Image.Skew', 'Image.Curt', 'Entropy'])
  # Use the XGBoost model to make a prediction
  predicted_class = xgb_model.predict(input_df)[0]
  if predicted_class == 0:
    output = 'Genuine'
  else:
    output = 'Counterfeit'
  return render_template('home_2.html', prediction=output)


app.run(host='0.0.0.0', port=8081)
