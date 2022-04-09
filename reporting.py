from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
import seaborn as sns


with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path'])


##############Function for reporting
def score_model():
    
    y_var, y_pred = model_predictions()

    cf_matrix = metrics.confusion_matrix(y_true=y_var, y_pred=y_pred)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    plt.savefig(os.path.join(model_path, "confusionmatrix.png"))





if __name__ == '__main__':
    score_model()
