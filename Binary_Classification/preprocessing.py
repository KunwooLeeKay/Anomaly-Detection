
def SMOTE(raw_data, feature):
    
    from imblearn.over_sampling import SMOTE

    X = raw_data[feature]

    Y1 = raw_data['고장 단계'] # 1번이 고장단계, 2번이 고장 유무
    Y2 = raw_data['고장 유무']

    sm = SMOTE(random_state = 10)

    X_binary, y_binary = sm.fit_resample(X, Y2) # 고장여부 판단 변수

    sm = SMOTE(k_neighbors = 2,random_state = 10)  # neighbor이 충분하지 않아서 oversampling을 못할때는 k를 2로 늘려주면 2번째로 가까운애로 사용하게 된다.

    X_multi, y_multi = sm.fit_resample(X, Y1) # 고장단계 판단 변수

    return X_binary, y_binary, X_multi, y_multi

def BorderlineSMOTE(raw_data, feature):
    from imblearn.over_sampling import BorderlineSMOTE

    X = raw_data[feature]

    Y1 = raw_data['고장 단계'] # 1번이 고장단계, 2번이 고장 유무
    Y2 = raw_data['고장 유무']

    sm = BorderlineSMOTE(random_state = 10)

    X_binary, y_binary = sm.fit_resample(X, Y2) # 고장여부 판단 변수

    sm = BorderlineSMOTE(k_neighbors = 2,random_state = 10)  # neighbor이 충분하지 않아서 oversampling을 못할때는 k를 2로 늘려주면 2번째로 가까운애로 사용하게 된다.

    X_multi, y_multi = sm.fit_resample(X, Y1) # 고장단계 판단 변수

    return X_binary, y_binary, X_multi, y_multi

def ADASYN(raw_data, feature):

    from imblearn.over_sampling import ADASYN

    X = raw_data[feature]

    Y1 = raw_data['고장 단계'] # 1번이 고장단계, 2번이 고장 유무
    Y2 = raw_data['고장 유무']

    sm = ADASYN(random_state = 10)

    X_binary, y_binary = sm.fit_resample(X, Y2) # 고장여부 판단 변수

    sm = ADASYN(k_neighbors = 2,random_state = 10)  # neighbor이 충분하지 않아서 oversampling을 못할때는 k를 2로 늘려주면 2번째로 가까운애로 사용하게 된다.

    X_multi, y_multi = sm.fit_resample(X, Y1) # 고장단계 판단 변수

    return X_binary, y_binary, X_multi, y_multi



import pandas as pd
raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")

feature = ['제조국가', '사용년수', 'C', 'D', 'E', '진동 횟수', '최대 기계압력', '최소 기계압력', '기계 길이',\
                    '기계 무게', '특수 부품 유무', '특수 부품 저항', '일반 부품 저항', '기계 마력', 'O', 'P',\
                    '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J',\
                    'L/J']

target = ['고장 단계', '고장 유무']

# 고장난게 90개밖에 안됨. 불균형 발생 -> oversampling 사용
X_binary_S, y_binary_S, X_multi_S, y_multi_S = SMOTE(raw_data, feature)
X_binary_BS, y_binary_BS, X_multi_BS, y_multi_BS = BorderlineSMOTE(raw_data, feature)
X_binary_A, y_binary_A, X_multi_A, y_multi_A = ADASYN(raw_data, feature)


# 데이터 Split
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X_binary_S, y_binary_S, random_state = 1, test_size = 0.3) # 고장 여부 예측
test_X, val_X, test_Y, val_Y = train_test_split(test_X, test_Y, random_state = 1, test_size = 0.5) # train / test / validation으로 데이터 분할



# 데이터 feature 간 차이가 큼 -> 정규화

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(raw_data[feature])
normalized_data = pd.DataFrame(X_scaled, columns = feature)  # 정규화된 learning data
normalized_data['고장 단계'] = raw_data['고장 단계']
normalized_data['고장 유무'] = raw_data['고장 유무']

# normalized_data.to_excel("NORMALIZED2.xlsx")

#######################

# from sklearn.model_selection import train_test_split

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score

# train_X, val_X, train_Y, val_Y = train_test_split(X_binary, y_binary, random_state = 1, test_size = 0.2) # 고장 여부 예측
# learn_model = RandomForestClassifier(random_state = 1)
# learn_model.fit(train_X, train_Y)

# val_predictions = learn_model.predict(val_X)
# print("고장 여부 validation", accuracy_score(val_Y, val_predictions))

# tn, fp, fn, tp = confusion_matrix(val_Y, val_predictions).ravel()
# specificity = tn / (tn + fp)
# sensitivity = tp / (tp + fn)
# print("위양성률 : {}, 위음성률 : {}".format(1- specificity, 1- sensitivity))


# train_X, val_X, train_Y, val_Y = train_test_split(X_multi, y_multi, random_state = 1, test_size = 0.2) # 고장 단계 예측
# learn_model = RandomForestClassifier(random_state = 1, max_features = 'sqrt')
# learn_model.fit(train_X, train_Y)

# val_predictions = learn_model.predict(val_X)
# print("고장 단계 validation", accuracy_score(val_Y, val_predictions))
