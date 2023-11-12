class Preprocessing:
    def __init__(self):
        import pandas as pd
        self.raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")
        self.feature = ['제조국가', '사용년수', 'C', 'D', 'E', '진동 횟수', '최대 기계압력', '최소 기계압력', '기계 길이',\
                        '기계 무게', '특수 부품 유무', '특수 부품 저항', '일반 부품 저항', '기계 마력', 'O', 'P',\
                        '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J',\
                        'L/J']
        self.target = ['고장 단계', '고장 유무']

    def B_Splitter(self):
        from sklearn.model_selection import train_test_split
        X = self.raw_data[self.feature]
        Y = self.raw_data['고장 유무']

        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state = None, test_size = 0.3, stratify = Y)
        test_X, val_X, test_Y, val_Y = train_test_split(test_X, test_Y, test_size=0.5, stratify = test_Y)
        
        return train_X, test_X, train_Y, test_Y, val_X, val_Y

    def M_Splitter(self):
        from sklearn.model_selection import train_test_split
        
        X = self.raw_data[self.feature]
        Y = self.raw_data['고장 단계'] # 1번이 고장단계, 2번이 고장 유무

        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state = None, test_size = 0.2, stratify = Y)
        
        return train_X, test_X, train_Y, test_Y

    def SMOTE(self, train_X, train_Y):
        from imblearn.over_sampling import SMOTE

        for k in [5, 4, 3, 2]:
            try:
                sm = SMOTE(random_state = None, k_neighbors = k)
                train_X, train_Y = sm.fit_resample(train_X, train_Y)
                print(k)
                break
            except:
                print("안됨")
                if k == 2:
                    print("오버샘플링 실패!")

        return train_X, train_Y

    def BorderlineSMOTE(self, train_X, train_Y):
        from imblearn.over_sampling import BorderlineSMOTE

        for k in [5, 4, 3, 2]:
            try:
                sm = BorderlineSMOTE(random_state = None, k_neighbors = k)
                train_X, train_Y = sm.fit_resample(train_X, train_Y)
                print(k)
                break
            except:
                print("안됨")
                if k == 2:
                    print("오버샘플링 실패!")

        return train_X, train_Y
    
    def ADASYN(self, train_X, train_Y):
        from imblearn.over_sampling import ADASYN

        for k in [5, 4, 3, 2]:
            try:
                sm = ADASYN(random_state = None, n_neighbors = k)
                print(k)
                train_X, train_Y = sm.fit_resample(train_X, train_Y)
                break
            except:
                print("안됨")
                if k == 2:
                    print("오버샘플링 실패!")

        return train_X, train_Y
    
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

train_X, test_X, train_Y, test_Y, val_X, val_Y = pp.B_Splitter()
# train_X, train_Y = pp.SMOTE(train_X, train_Y)
# train_X, test_X = pp.Normalize(train_X, test_X)
train_X, test_X = pp.Normalize(train_X, test_X)


import Module_LogisticRegression_New as LR
from imblearn.over_sampling import BorderlineSMOTE

model, param, scores = LR.LogisticRegressor(train_X, train_Y, BorderlineSMOTE())

train_pred = model.predict(train_X)
val_pred = model.predict(val_X)
test_pred = model.predict(test_X)

from sklearn.metrics import confusion_matrix # 가로가 진짜!
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
print(confusion_matrix(train_Y, train_pred))
print(confusion_matrix(val_Y, val_pred))
print(confusion_matrix(test_Y, test_pred))

print(classification_report(test_Y, test_pred))
print(scores['mean_test_score'])