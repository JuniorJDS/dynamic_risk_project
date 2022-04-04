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
model_path = os.path.join(config['output_model_path']) 


def preprocess_data(dataframe: pd.DataFrame):
    
    y_var = dataframe["exited"]
    X_var = dataframe.drop(
        ["exited", "corporation"], axis=1
    )

    return X_var, y_var


def train_model():

    dataframe = pd.read_csv(os.path.join(
        dataset_csv_path, "finaldata.csv"
    ))
    
    X_var, y_var = preprocess_data(dataframe)
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_var, y_var, test_size=0.20
    )

    # use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='ovr', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    # fit the logistic regression to your data
    model.fit(X_train, y_train)
    
    # saving the trained model
    model_dest_path = os.path.join(
        model_path, "trainedmodel.pkl"
    )
    with open(model_dest_path, "wb") as model_file:
        pickle.dump(model, model_file)


if __name__ == '__main__':
    train_model()