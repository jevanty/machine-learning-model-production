from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from .preprocessors import NanDealing, FeatureEngineering, GetDummies

prediction_pipeline = Pipeline([
    ('Deal with NaN', NanDealing()),
    ('Feature engineering', FeatureEngineering()),
    ('Dummies on categorical variables', GetDummies()),
    ('model creation', RandomForestClassifier(criterion='gini', n_estimators=700,
                                              min_samples_split=10, min_samples_leaf=1,
                                              max_features='auto', oob_score=True,
                                              random_state=1, n_jobs=-1))
])
