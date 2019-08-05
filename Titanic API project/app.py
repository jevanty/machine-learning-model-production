import pandas as pd
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from sqlalchemy import create_engine
from create_db import export_csv_to_db
import os

#Init app
app = Flask(__name__)
#basedir = os.path.abspath(os.path.dirname(__file__))
DATABASE_URL = os.environ['DATABASE_URL']
#DATABASE_URL = 'postgres://zrqhgahz:XVI1tVLrMvMMOYK_QgQUhBRr9qp8n9Zr@manny.db.elephantsql.com:5432/zrqhgahz'

# O.R.M.:
engine = create_engine(DATABASE_URL)
if os.environ['IS_DB_FILLED'] == 'no':
    export_csv_to_db(engine)

#Database
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from ML_MODEL.prediction import make_prediction
from ML_MODEL.train import train_model

#Init db
db = SQLAlchemy(app)

#Init ma
ma = Marshmallow(app)

# routes

@app.route('/predict', methods=['POST'])
def predict():
    y_pred = make_prediction(input_data=request.get_json())
    df = pd.DataFrame(request.get_json(), index=[0])
    df['Survived'] = y_pred
    df.to_sql('predictions', con=engine, if_exists='append')
    return jsonify(y_pred.tolist())

@app.route('/createTableTest',methods=['POST'])
def createDatabaseTest():
  data = pd.read_csv(request.files.get('test_file'), index_col=0)
  data.to_sql('passengers_test', con=engine)
  return "Test table has been created"

@app.route('/predictTest',methods=['GET'])
def test():
  dataToTest = pd.read_sql('passengers_test', con=engine)
  y_predTest = make_prediction(dataToTest).tolist()
  dataToTest['Survived'] = y_predTest
  dataToTest.to_sql('passengers_test_predictions', con=engine, if_exists='append',index=False)
  return "Predictions on test Table has been filled up"

@app.route('/data', methods=['GET'])
def data():
    return pd.read_sql('passengers', DATABASE_URL).to_dict()

@app.route('/train', methods=['GET'])
def train():
    train_model()
    return "Training completed"

@app.route('/deleteTable',methods=['DELETE'])
def deleteTable():
  try:
    db.engine.execute('DROP TABLE passengers')
    return "Your table has been deleted"
  except:
    return "No table"

@app.route('/create', methods=['GET'])
def db_():
    export_csv_to_db(engine)
    return "Your new table has been created"


#Run server
if __name__=='__main__':
    app.run(host='0.0.0.0', port=os.environ['PORT'])