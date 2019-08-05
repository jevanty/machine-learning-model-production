import pandas as pd
import os
from sklearn.model_selection import train_test_split
from .pipeline_preparation import prediction_pipeline
import pickle
from sqlalchemy import create_engine

#print("real", os.path.realpath(__file__))

url = os.environ['DATABASE_URL']
#url = 'postgres://zrqhgahz:XVI1tVLrMvMMOYK_QgQUhBRr9qp8n9Zr@manny.db.elephantsql.com:5432/zrqhgahz'
engine = create_engine(url)
TRAINING_DATA_FRAME = pd.read_sql_table('passengers', con=engine)

MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/saved_models/model.sav'

# TRAINING_DATA_FRAME = os.path.dirname(os.path.realpath(__file__)) + '/data/train.csv'
# TESTING_DATA_FILE = os.path.dirname(os.path.realpath(__file__)) + '/data/test.csv'


def _save_model(model):
    pickle.dump(model, open(MODEL_PATH, 'wb'))


def train_model():
    data = TRAINING_DATA_FRAME
    data.dropna(subset=['Survived'], inplace=True)
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    prediction_pipeline.fit(X_train, y_train)
    print(prediction_pipeline.score(X_test, y_test))
    _save_model(prediction_pipeline)

if __name__ == '__main__':
    print(train_model())