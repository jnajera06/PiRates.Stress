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
    extract data from html options choosen for the users
    '''
    m = request.form['model']
    Q10_1 = int(request.form['Q10_1'])
    Q10_4 = int(request.form['Q10_4'])
    Q10_5 = int(request.form['Q10_5'])
    Q10_6 = int(request.form['Q10_6'])
    Q10_7 = int(request.form['Q10_7'])
    Q10_9 = int(request.form['Q10_9'])
    Q10_12 = int(request.form['Q10_12'])
    Q17_1 = int(request.form['Q17_1'])
    Q17_7 = int(request.form['Q17_7'])
    Q17_10 = int(request.form['Q17_10'])

        
    
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
    
    final_features = [[Q10_1,Q10_4,Q10_5,Q10_6,Q10_7,Q10_9,Q10_12,Q17_1,Q17_7,Q17_10]]
    
    model_knn = pickle.load(open('K_nearest_Neighbors_Classifier_model.pkl', 'rb'))
    pred_knnm = model_knn.predict(final_features)
    output_knn = round(pred_knnm[0], 2)
    
    model_dt = pickle.load(open('Decision_tree_model.pkl', 'rb'))
    pred_dtm = model_dt.predict(final_features)
    output_dt = round(pred_dtm[0], 2)
    
    model_rf = pickle.load(open('random_forest_Classifier_model.pkl', 'rb'))
    pred_rfm = model_rf.predict(final_features)
    output_rf = round(pred_rfm[0], 2)
    
    model_nn = pickle.load(open('Neural_network_model.pkl', 'rb'))
    pred_nnm = model_nn.predict(final_features)
    output_nn = round(pred_nnm[0], 2)
    
    model_stack = pickle.load(open('Stack_model.pkl', 'rb'))
    pred_stack = model_stack.predict(final_features)
    output_stack = round(pred_stack[0], 2)
    
    if m == "1":
      model_sel = "K nearest neighbors "
      model = model_knn
    elif m == "2":
      model_sel = "Decision tree "
      model = model_dt
    elif m == "3":
      model_sel = "Random forest "
      model = model_rf
    elif m == "4":
      model_sel = "Neural network "
      model = model_nn
    elif m == "5":
      model_sel = "Stack "
      model = model_stack
    else:
      model = model_rf
      
    
    '''
    For rendering results on HTML GUI
    '''
    
    prediction = model.predict(final_features)
    #prediction = model.predict([[Q10_1,Q10_4,Q10_5,Q10_6,Q10_7,Q10_9,Q10_12,Q17_1,Q17_7,Q17_10]])     
    output = round(prediction[0], 2)

    return render_template('prediction.html', prediction_text= 'Based on model '+model_sel+ 'your stress Level is:   {} out of 5'.format(output),
                           pred_knn ='stress Level {} out of 5'.format(output_knn),
                           pred_dt ='stress Level {} out of 5'.format(output_dt),
                           pred_rf ='stress Level {} out of 5'.format(output_rf),
                           pred_nn ='stress Level {} out of 5'.format(output_nn),
                           pred_stack ='stress Level {} out of 5'.format(output_stack))
    


if __name__ == "__main__":
    app.run(debug=False)# -*- coding: utf-8 -*-
    
   

