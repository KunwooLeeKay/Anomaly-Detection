
import pickle

with open('feature.pickle', 'rb') as pickle_data:
    features = pickle.load(pickle_data)

import pandas as pd
raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")


from sklearn.preprocessing import StandardScaler


from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score, f1_score

ftwo_scorer = make_scorer(fbeta_score, beta = 2)
fone_scorer = make_scorer(f1_score)

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

model = Pipeline([
    ('sampling', SMOTE()),
    ("model", LogisticRegression(max_iter = 1000))
])

from sklearn.model_selection import cross_val_score

qualitative_data = ['제조국가', 'C', 'D', 'E', '특수 부품 유무']
qualitative_data = ['C', 'D', 'E', '특수 부품 유무']
qualitative_data = ['D']



for feature in features:
    feature.extend(qualitative_data)
    # print(feature, end = ' : ')
    X, Y = raw_data[feature], raw_data['고장 유무']
    standard_scaler = StandardScaler()
    X = standard_scaler.fit_transform(X)
    scores = cross_val_score(model, X, Y, scoring = ftwo_scorer, cv = 10)
    score = sum(scores)/10
    if score > 0.4:
        print(feature, score, sep = ' : ')

