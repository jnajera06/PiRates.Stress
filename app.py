import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
#model = pickle.load(open('random_forest_Classifier_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    '''
    extract the model choosen
    '''
    m = request.form["model"]
        
    if m == '1':
      model_sel = "K nearest neighbors "
      model = pickle.load(open('K_nearest_Neighbors_Classifier_model.pkl', 'rb'))
    elif m == '2':
      model_sel = "Decision tree "
      model = pickle.load(open('Decision_tree_model.pkl', 'rb'))
    elif m == '3':
      model_sel = "Random forest "
      model = pickle.load(open('random_forest_Classifier_model.pkl', 'rb'))
    elif m == '4':
      model_sel = "Neural network "
      model = pickle.load(open('Neural_network_model.pkl', 'rb'))
    else:
      model_sel = "Stack "
      model = pickle.load(open('Stack_model.pkl', 'rb'))
      
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
  
    
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text= 'Based on model '+model_sel+ 'your stress Level is:   {} out of 5'.format(output))
    


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    model = pickle.load(open('random_forest_Classifier_model.pkl', 'rb'))
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)# -*- coding: utf-8 -*-

