from flask import Flask, session, jsonify, request
import json
import os
from diagnostics import (
    model_predictions, 
    dataframe_summary, 
    missing_data, 
    execution_time, 
    outdated_packages_list
)
from scoring import score_model



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    dataset_path = request.json.get('dataset_path')
    _, y_pred = model_predictions(dataset_path)
    return y_pred


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    score = score_model()
    return str(score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    summary = dataframe_summary()
    return summary

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    execution_time = execution_time()
    missing_data = missing_data()
    outdated_packages_list = outdated_packages_list()     
    return str(
        "execution_time:" + 
        execution_time + "\nmissing_data;"+ 
        missing_data + "\noutdated_packages:" + 
        outdated_packages_list
    )


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
