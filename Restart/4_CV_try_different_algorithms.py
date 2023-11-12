
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

#SAMPLING#######################################################################################

###Oversampling###
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, ADASYN

smote, b_smote, s_smote, k_smote, adasyn = SMOTE(), BorderlineSMOTE(), SVMSMOTE(), KMeansSMOTE(), ADASYN()

over_sampling = [smote, b_smote, s_smote, k_smote, adasyn]

###OVER + UnderSampling###
from imblearn.combine import SMOTETomek, SMOTEENN
smotetomek, smoteenn = SMOTETomek(), SMOTEENN()

combine_sampling = [smotetomek, smoteenn]

samplers = [smote, b_smote, s_smote, k_smote, adasyn, smotetomek, smoteenn]
sampler_names = ['SMOTE', 'BorderlineSMOTE', 'SVMSMOTE', 'KMeansSMOTE', 'ADASYN', 'SMOTETomek', 'SMOTEENN']

#ALGORITHMS#######################################################################################

### LINEAR MODEL ###
from sklearn.linear_model import LogisticRegression, RidgeClassifier

lr = LogisticRegression(max_iter=5000)
rc = RidgeClassifier()

linear_models = [lr, rc]
linear_names = ['LogisticRegression', 'RidgeClassifier']

### NONLINEAR MODEL ###
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

dtc = DecisionTreeClassifier()
knn = KNeighborsClassifier()
svm = SVC()
ann = MLPClassifier()
gpc = GaussianProcessClassifier()

nonlinear_models = [dtc, knn, svm, ann, gpc]
nonlinear_names = ['DecisionTreeClassifier', 'KNeighborsClassifier', 'SVC', 'MLPClassifier', 'GaussianProcessClassifier']

### ENSEMBLE MODEL ###
from sklearn.ensemble import RandomForestClassifier #, StackingClassifier, VotingClassifier : 추후 추가
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

rfc = RandomForestClassifier()
xgbm, lgbm = XGBClassifier(use_label_encoder=False), LGBMClassifier()

ensemble_models = [rfc, xgbm, lgbm]
ensemble_names = ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline


# COMPARISON WITH SAMPLING + SCALER ###################################################################
scalers = ['Normalize', 'Standardize']

whole_scores = {}
sampler_scores = {}
scaler_scores = {}
temp_scores = []

for learn_model, model_name in zip(linear_models, linear_names):
    scaler_scores = {}
    for scaler in scalers:
        print(scaler)
        sampler_scores = {}
        for sampler, sampler_name in zip(samplers, sampler_names):
            if sampler_name == 'KMeansSMOTE':
                continue
            print(sampler)
            
            temp_scores.clear()

            for random_seed in range(10):
                train_X, test_X, train_Y, test_Y = pp.B_Splitter(random_seed)

                if scaler == 'Normalize':
                    train_X, test_X = pp.Normalize(train_X, test_X)
                elif scaler == 'Standardize':
                    train_X, test_X = pp.Standardize(train_X, test_X)

                model = Pipeline([
                    ('sampling', sampler),
                    ('classifier', learn_model)
                ])

                scores = cross_val_score(model, train_X, train_Y, scoring = ftwo_scorer, cv = 10)
                score = sum(scores)/10
                
                temp_scores.append(score)
            
            # print(temp_scores)
            score = sum(temp_scores)/10
            print(score)
            sampler_scores[sampler_name] = score
        
        scaler_scores[scaler] = sampler_scores
        print("scaler",scaler_scores)
    whole_scores[model_name] = scaler_scores
    
sampler_scores = {}
scaler_scores = {}
temp_scores = []
for learn_model, model_name in zip(nonlinear_models, nonlinear_names):
    scaler_scores = {}
    for scaler in scalers:
        print(scaler)
        sampler_scores = {}
        for sampler, sampler_name in zip(samplers, sampler_names):
            if sampler_name == 'KMeansSMOTE':
                continue
            print(sampler)
            
            temp_scores.clear()

            for random_seed in range(10):
                train_X, test_X, train_Y, test_Y = pp.B_Splitter(random_seed)

                if scaler == 'Normalize':
                    train_X, test_X = pp.Normalize(train_X, test_X)
                elif scaler == 'Standardize':
                    train_X, test_X = pp.Standardize(train_X, test_X)

                model = Pipeline([
                    ('sampling', sampler),
                    ('classifier', learn_model)
                ])

                scores = cross_val_score(model, train_X, train_Y, scoring = ftwo_scorer, cv = 10)
                score = sum(scores)/10
                
                temp_scores.append(score)
            
            # print(temp_scores)
            score = sum(temp_scores)/10
            print(score)
            sampler_scores[sampler_name] = score
        
        scaler_scores[scaler] = sampler_scores
        print("scaler",scaler_scores)
    whole_scores[model_name] = scaler_scores
    
sampler_scores = {}
scaler_scores = {}
temp_scores = []
for learn_model, model_name in zip(ensemble_models, ensemble_names):
    scaler_scores = {}
    for scaler in scalers:
        print(scaler)
        sampler_scores = {}
        for sampler, sampler_name in zip(samplers, sampler_names):
            if sampler_name == 'KMeansSMOTE':
                continue
            print(sampler)
            
            temp_scores.clear()

            for random_seed in range(10):
                train_X, test_X, train_Y, test_Y = pp.B_Splitter(random_seed)

                if scaler == 'Normalize':
                    train_X, test_X = pp.Normalize(train_X, test_X)
                elif scaler == 'Standardize':
                    train_X, test_X = pp.Standardize(train_X, test_X)

                model = Pipeline([
                    ('sampling', sampler),
                    ('classifier', learn_model)
                ])

                scores = cross_val_score(model, train_X, train_Y, scoring = ftwo_scorer, cv = 10)
                score = sum(scores)/10
                
                temp_scores.append(score)
            
            # print(temp_scores)
            score = sum(temp_scores)/10
            print(score)
            sampler_scores[sampler_name] = score
        
        scaler_scores[scaler] = sampler_scores
        print("scaler",scaler_scores)
    whole_scores[model_name] = scaler_scores
    


import pickle
with open("Select_Algorithm.pickle", 'wb') as pickle_data:
    pickle.dump(whole_scores, pickle_data)