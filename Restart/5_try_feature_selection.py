
class Preprocessing:
    def __init__(self):
        import pandas as pd
        self.raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")
        self.feature = ['제조국가', '사용년수', 'C', 'D', 'E', '진동 횟수', '최대 기계압력', '최소 기계압력', '기계 길이',\
                        '기계 무게', '특수 부품 유무', '특수 부품 저항', '일반 부품 저항', '기계 마력', 'O', 'P',\
                        '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J',\
                        'L/J']
        self.target = ['고장 단계', '고장 유무']

    def B_Splitter(self, seed):
        from sklearn.model_selection import train_test_split
        X = self.raw_data[self.feature]
        Y = self.raw_data['고장 유무']

        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state = seed, test_size = 0.2, stratify = Y)
        
        return train_X, test_X, train_Y, test_Y

    
    def Normalize(self, train_X, test_X):

        from sklearn.preprocessing import MinMaxScaler

        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(train_X)

        train_X_scaled = min_max_scaler.transform(train_X)
        test_X_scaled = min_max_scaler.transform(test_X)

        return train_X_scaled, test_X_scaled

    def Standardize(self, train_X, test_X):

        from sklearn.preprocessing import StandardScaler

        standard_scaler = StandardScaler()
        standard_scaler.fit(train_X)

        train_X_scaled = standard_scaler.transform(train_X)
        test_X_scaled = standard_scaler.transform(test_X)

        return train_X_scaled, test_X_scaled

pp = Preprocessing()

from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score

ftwo_scorer = make_scorer(fbeta_score, beta = 2)

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

train_X, test_X, train_Y, test_Y = pp.B_Splitter(1)
train_X, test_X = pp.Standardize(train_X, test_X)

sel = SelectFromModel(estimator = LogisticRegression(max_iter=5000))



model = Pipeline([
    ('sampling', SMOTE()),
    ('classifier', sel)
])

from sklearn.model_selection import cross_val_score

score = cross_val_score(model, train_X, train_Y, scoring = ftwo_scorer, cv = 10)
print(sum(score)/10)