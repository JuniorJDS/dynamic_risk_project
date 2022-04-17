
import os
import subprocess
import json
import timeit
import sys

from joblib import load

import pandas as pd



##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['prod_deployment_path'])


def preprocess_data(dataframe: pd.DataFrame):
    
    y_var = dataframe["exited"]
    X_var = dataframe.drop(
        ["exited", "corporation"], axis=1
    )

    return X_var, y_var


##################Function to get model predictions
def model_predictions(dataset_path: str = None):

    model = load(os.path.join(model_path, "trainedmodel.pkl"))

    if not dataset_path:  dataset_path = "testdata.csv"

    dataframe = pd.read_csv(os.path.join(test_data_path, dataset_path))
    X_var, y_var = preprocess_data(dataframe)

    y_pred = model.predict(X_var) 

    return y_var, y_pred

##################Function to get summary statistics
def dataframe_summary():

    dataframe = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    numeric_columns = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees"
        ]
    
    result = []
    for column in numeric_columns:
        summary_statistics = {
            "column": column,
            "mean": dataframe[column].mean(),
            "median": dataframe[column].median(),
            "standard deviation": dataframe[column].std()
        }
        result.append(summary_statistics)
    return result

def missing_data():
    dataframe = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    result = []
    for column in dataframe.columns:
        count_na = dataframe[column].isna().sum()
        count_not_na = dataframe[column].count()
        count_total = count_not_na + count_na
        result.append(
            {
            "column": column,
            "percent": str(int(count_na/count_total*100))+"%"
            }
        )
    return result

##################Function to get timings
def execution_time():

    result = []
    for procedure in ["training.py" , "ingestion.py"]:
        starttime = timeit.default_timer()
        os.system('python3 %s' % procedure)
        timing=timeit.default_timer() - starttime
        result.append(
            {
                "task": procedure,
                "time": timing
            }
        )
 
    return result

##################Function to check dependencies
def outdated_packages_list():
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    
    return str(outdated_packages)

if __name__ == '__main__':
    model_predictions(dataset_path=None)
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()





    
