import pandas as pd
import numpy as np
import os
import json
from glob import glob
from datetime import datetime


with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe():

    csv_files = glob(f"{input_folder_path}/*.csv")

    dataframe_concated = pd.concat(
        map(pd.read_csv, csv_files), ignore_index=True
    )

    dataframe_concated.drop_duplicates(inplace=True)

    dataframe_concated.to_csv(
        f"{output_folder_path}/finaldata.csv", index=False
    )

    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w") as report_file:
        for line in csv_files:
            report_file.write(line + "\n")



if __name__ == '__main__':
    merge_multiple_dataframe()
