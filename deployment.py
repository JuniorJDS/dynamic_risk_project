from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from shutil import copy2


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path']) 


def store_model_into_pickle():
    
    for file in [
        "ingestedfiles.txt", 
        "trainedmodel.pkl", 
        "latestscore.txt"
    ]:
        if file == "ingestedfiles.txt":
            src_filepath = os.path.join(dataset_csv_path, file)
        else:
            src_filepath = os.path.join(model_path, file)

        dest_filepath = os.path.join(prod_deployment_path, file)
        print(f'Copying {src_filepath} to {dest_filepath}')
        copy2(src_filepath, dest_filepath)


if __name__ == '__main__':
    store_model_into_pickle()
        
        

