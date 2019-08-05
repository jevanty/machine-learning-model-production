import pickle
from .train import MODEL_PATH
import pandas as pd


def make_prediction(input_data, model_path=MODEL_PATH):
    if type(input_data) is dict:
        data = pd.DataFrame(input_data, index=[0])
    else:
        data = pd.DataFrame(input_data)

    loaded_model = pickle.load(open(model_path, 'rb'))
    # X = test_data.drop('survived', axis=1, errors='ignore')
    result = loaded_model.predict(data)
    return result
