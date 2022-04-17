import requests
import os
import json


def main():

    URL = "http://127.0.0.1:5000"
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    response1 = requests.post(
        f"{URL}/prediction", 
        json={"dataset_path": "testdata.csv"}, headers=headers).text
    response2 = requests.get(f"{URL}/scoring", headers=headers).text
    response3 = requests.get(f"{URL}/summarystats", headers=headers).text
    response4 = requests.get(f"{URL}/diagnostics", headers=headers).text


    # combine all API responses
    responses = response1 + "\n" + response2 + "\n" + response3 + "\n" + response4

    # write the responses to your workspace
    with open('config.json','r') as f:
        config = json.load(f) 
    model_path = os.path.join(config['output_model_path'])

    with open(os.path.join(model_path, "apireturns.txt"), "w") as returns_file:
        returns_file.write(responses)


if __name__ == "__main__":
    main()