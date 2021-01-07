# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:38:02 2020

@author: aboyite
"""


from flask import Flask, request
from pandas import DataFrame
from datetime import datetime
from flasgger import Swagger
import joblib

app = Flask(__name__)# start the app __name__ specfiy the which particular point application should start
Swagger(app)

local_path_pickle = r"C:\Users\aboyite\Desktop\Deep Learning POC\Electricity demand forecasting\Electricity_demand_forecasting_deployment\ElectricityForcasting.pkl"
# Load the model from the file 
prophet_model = joblib.load(local_path_pickle)  

#pickle_in = open("ElectricityForcasting.pkl","rb")
#prophet_model=pickle.load(pickle_in)
 
@app.route("/")
def start_page():
    return "Welcome ALL"

@app.route("/predicition", methods=["Get"])
def predict_Electrcity_demand():
    
    """Application for Spain Electricity Demand 
    ---
    parameters:
      - name: date
        in: query
        type: string
        required: true
    responses:
        200:
            description: The output values
            
    """
    date_list = []
    
    date_form = request.args.get("date")
    print(date_form)
     
    date_dt = datetime.strptime(date_form, '%d-%m-%Y')
    date_list.append(date_dt)
    test_data = DataFrame(date_list, columns=["ds"])
    
    prediction_result = prophet_model.predict(test_data)
    print(prediction_result.iloc[0]["yhat"])
    return "The predicted values is:" + str(prediction_result.iloc[0]["yhat"])
 
 
if __name__ == "__main__":
    app.run()
 
 
 