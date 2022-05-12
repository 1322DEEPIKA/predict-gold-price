import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

df = pd.read_csv('gold_rate_history.csv')
x=df.drop(['Date'],axis=1)

y=df['Date']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [int(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features) 

    if prediction <= 3000:
        pred = "GOLD RATE IS HIGH THEN 6K."
    elif prediction == 6000:
        pred = "GOLD RATE IS LESS THEN 3K."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
