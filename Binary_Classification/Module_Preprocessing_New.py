
# feature =  ['제조국가', '사용년수', 'C', 'D', 'E', '진동 횟수', '최대 기계압력', '최소 기계압력', '기계 길이',\
#                         '기계 무게', '특수 부품 유무', '특수 부품 저항', '일반 부품 저항', '기계 마력', 'O', 'P',\
#                         '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J',\
#                         'L/J']
# remove_feature = ['제조국가', 'C', 'D', 'E', '특수 부품 유무']


# for i in remove_feature:
#     feature.remove(i) 

# print(feature)

class Preprocessing:
    def __init__(self):
        import pandas as pd
        self.raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")
        self.feature = ['제조국가', '사용년수', 'C', 'D', 'E', '진동 횟수', '최대 기계압력', '최소 기계압력', '기계 길이',\
                        '기계 무게', '특수 부품 유무', '특수 부품 저항', '일반 부품 저항', '기계 마력', 'O', 'P',\
                        '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J',\
                        'L/J']
        # self.feature = feature
        
        self.target = ['고장 단계', '고장 유무']

    def B_Splitter(self):
        from sklearn.model_selection import train_test_split
        X = self.raw_data[self.feature]
        Y = self.raw_data['고장 유무']

        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state = None, test_size = 0.2, stratify = Y)
        
        return train_X, test_X, train_Y, test_Y

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
