
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

pp = Preprocessing()
train_X, test_X, train_Y, test_Y = pp.B_Splitter()

#ALGORITHMS#######################################################################################

### LINEAR MODEL ###
from sklearn.linear_model import LogisticRegression, RidgeClassifier

lr = LogisticRegression()
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
xgbm, lgbm = XGBClassifier(), LGBMClassifier()

ensemble_models = [rfc, xgbm, lgbm]
ensemble_names = ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']

#SAMPLING#######################################################################################

###Oversampling###
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, ADASYN

smote, b_smote, s_smote, k_smote, adasyn = SMOTE(), BorderlineSMOTE(), SVMSMOTE(), KMeansSMOTE(), ADASYN()

over_sampling = [smote, b_smote, s_smote, k_smote, adasyn]

###OVER + UnderSampling###
from imblearn.combine import SMOTETomek, SMOTEENN
smotetomek, smoteenn = SMOTETomek(), SMOTEENN()

combine_sampling = [smotetomek, smoteenn]

sampling = [smote, b_smote, s_smote, k_smote, adasyn, smotetomek, smoteenn]
sampler_name = ['SMOTE', 'BorderlineSMOTE', 'SVMSMOTE', 'KMeansSMOTE', 'ADASYN', 'SMOTETomek', 'SMOTEENN']


from sklearn.metrics import fbeta_score
import warnings
warnings.filterwarnings('ignore')


# COMPARISON WITHOUT SAMPLING ###################################################################

# from imblearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer

# ftwo_scorer = make_scorer(fbeta_score, beta = 2)

for num in range(10):
    parameters = {}

    dic = {}
    score = {}

    print("No Sampling")

    for learn_model, name in zip(linear_models, linear_names):
        learn_model.fit(train_X, train_Y)
        print("linear model : ",name, fbeta_score(test_Y, learn_model.predict(test_X), beta = 2))
        score[name] = fbeta_score(test_Y, learn_model.predict(test_X), beta = 2)
        

    for learn_model, name in zip(nonlinear_models, nonlinear_names):
        learn_model.fit(train_X, train_Y)
        print("nonlinear model : ",name, fbeta_score(test_Y, learn_model.predict(test_X), beta = 2))
        score[name] = fbeta_score(test_Y, learn_model.predict(test_X), beta = 2)

    for learn_model, name in zip(ensemble_models, ensemble_names):
        learn_model.fit(train_X, train_Y)
        print("emsemble model : ",name, fbeta_score(test_Y, learn_model.predict(test_X), beta = 2))
        score[name] = fbeta_score(test_Y, learn_model.predict(test_X), beta = 2)

    dic['No Sampling'] = score
    import pandas as pd
    df = pd.DataFrame(dic)
    df.to_excel("No_Sampling_Result"+str(num)+".xlsx")

    # COMPARISON WITH SAMPLING ###################################################################

    train_X, test_X, train_Y, test_Y = pp.B_Splitter()

    dic = {}


    print("\n########\nYES Sampling")

    for sampler, samp_name in zip(sampling, sampler_name):
        print('Sampler : ', samp_name)
        train_X, train_Y = sampler.fit_resample(train_X, train_Y)
        score = {}

        for learn_model, name in zip(linear_models, linear_names):
            learn_model.fit(train_X, train_Y)
            print("linear model : ", name ,fbeta_score(test_Y, learn_model.predict(test_X), beta = 2))
            score[name] = fbeta_score(test_Y, learn_model.predict(test_X), beta = 2)

        for learn_model, name in zip(nonlinear_models, nonlinear_names):
            learn_model.fit(train_X, train_Y)
            print("nonlinear model : ",name, fbeta_score(test_Y, learn_model.predict(test_X), beta = 2))
            score[name] = fbeta_score(test_Y, learn_model.predict(test_X), beta = 2)

        for learn_model, name in zip(ensemble_models, ensemble_names):
            learn_model.fit(train_X, train_Y)
            print("emsemble model : ",name, fbeta_score(test_Y, learn_model.predict(test_X), beta = 2))
            score[name] = fbeta_score(test_Y, learn_model.predict(test_X), beta = 2)
        
        dic[samp_name] = score

    df = pd.DataFrame(dic)
    df.to_excel("Yes_Sampling_Result"+str(num)+".xlsx")

    # COMPARISON WITH SAMPLING + SCALER ###################################################################



    dic = {}
    final_df = pd.DataFrame

    train_X, test_X, train_Y, test_Y = pp.B_Splitter()

    scaler = [pp.Normalize(train_X, test_X), pp.Standardize(train_X, test_X)]

    print("\n########\n Sampling + Scaling")

    for scaler_name in ['Normalize', 'Standardize']:
        print(' ##SCALER## : ', scaler_name)

        sub_dic = {}

        for sampler, samp_name in zip(sampling, sampler_name):
            print('Sampler : ', samp_name)
            train_X, train_Y = sampler.fit_resample(train_X, train_Y)

            score = {}

            if scaler_name == 'Normalize':
                train_X, test_X = pp.Normalize(train_X, test_X)
            elif scaler_name == 'Standardize':
                train_X, test_X = pp.Standardize(train_X, test_X)

            for learn_model, name in zip(linear_models, linear_names):
                learn_model.fit(train_X, train_Y)
                print("linear model : ", name ,fbeta_score(test_Y, learn_model.predict(test_X), beta = 2))
                score[name] = fbeta_score(test_Y, learn_model.predict(test_X), beta = 2)

            for learn_model, name in zip(nonlinear_models, nonlinear_names):
                learn_model.fit(train_X, train_Y)
                print("nonlinear model : ",name, fbeta_score(test_Y, learn_model.predict(test_X), beta = 2))
                score[name] = fbeta_score(test_Y, learn_model.predict(test_X), beta = 2)

            for learn_model, name in zip(ensemble_models, ensemble_names):
                learn_model.fit(train_X, train_Y)
                print("emsemble model : ",name, fbeta_score(test_Y, learn_model.predict(test_X), beta = 2))
                score[name] = fbeta_score(test_Y, learn_model.predict(test_X), beta = 2)

            sub_dic[samp_name] = score

            df = pd.DataFrame(sub_dic)
            df.to_excel("Scaling_Sampling_Result_"+str(scaler_name)+ "_" + str(num)+".xlsx")