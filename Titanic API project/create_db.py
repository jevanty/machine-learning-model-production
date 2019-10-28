import pandas as pd

def export_csv_to_db(engine):
    df = pd.read_csv('ML_MODEL/data/train.csv', index_col=0)
    df.to_sql('passengers', con=engine)

if __name__== '__main__':
    export_csv_to_db()




