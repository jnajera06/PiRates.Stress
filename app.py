import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('random_forest_Classifier_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    '''
    extract the model choosen
    '''
    m = request.form["model"]
    
    if m == '1':
      model_sel = "K nearest neighbors "
    elif m == '2':
      model_sel = "Decision tree "
    elif m == '3':
      model_sel = "Random forest "
    elif m == '4':
      model_sel = "Neural network "
    else:
      model_sel = "Stack "
    
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text= 'Based on model '+model_sel+ 'your stress Level is:   {} out of 5'.format(output))
    


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)# -*- coding: utf-8 -*-

