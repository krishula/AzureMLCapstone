import json
import numpy as np
import os
import joblib
import pandas as pd


def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), './heart-diesease-auto-model.pkl')
    #model_path = '.outputs/heart-diesease-auto-model.pkl'
    model = joblib.load(model_path)

def run(data):
    try:
        #data = np.array(json.loads(data))
        data = json.loads(data)['data']
        data = pd.DataFrame.from_dict(data)
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error

