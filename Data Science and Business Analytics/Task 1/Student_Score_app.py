# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:58:55 2022

@author: dell
"""

from flask import Flask, render_template, request
import joblib
import numpy as np

app= Flask(__name__)
Model=joblib.load('Student_Score_Model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    #hour=5.9 Hours
    Hours=request.form.get('Hours')
    #print(Hours)
    #Hours=request.form.values()
    #Score=Model.predict(np.array(hour).reshape(-1,1)).flatten()[0]
    Score=Model.predict(np.array(Hours).reshape(-1,1)).flatten()[0]

    return render_template("index.html",prediction_text=np.round(Score,2))
        
if __name__ == '__main__':
   app.run()
    