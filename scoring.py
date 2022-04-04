from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path']) 


def preprocess_data(dataframe: pd.DataFrame):
    
    y_var = dataframe["exited"]
    X_var = dataframe.drop(
        ["exited", "corporation"], axis=1
    )

    return X_var, y_var


def score_model():

    model_dest_path = os.path.join(
        model_path, "trainedmodel.pkl"
    )

    with open(model_dest_path, "rb") as model_file:
        model = pickle.load(model_file)

    dataframe = pd.read_csv(
        os.path.join(test_data_path, "testdata.csv")
    )

    X_var, y_var = preprocess_data(dataframe)

    y_pred = model.predict(X_var)

    # calculating F1 score of trained model
    f1_score = metrics.f1_score(y_var, y_pred)

    # writing F1 score to a file
    with open(os.path.join(model_path, "latestscore.txt"), "w") as score_file:
        score_file.write(str(f1_score) + "\n")

    return f1_score
    


if __name__ == '__main__':
    score_model()
