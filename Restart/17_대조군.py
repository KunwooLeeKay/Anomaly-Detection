import pandas as pd
from sklearn.model_selection import cross_val_predict
raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")
feature = ['제조국가', '사용년수', 'C', 'D', 'E', '진동 횟수', '최대 기계압력', '최소 기계압력', '기계 길이',\
                        '기계 무게', '특수 부품 유무', '특수 부품 저항', '일반 부품 저항', '기계 마력', 'O', 'P',\
                        '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J',\
                        'L/J']


def Evaluate(data, feature):

    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    X = data[feature]
    Y = data['고장 유무']

    standard_scaler = StandardScaler()
    standard_scaler.fit(X)

    X = standard_scaler.transform(X)

    model = Pipeline([
        ('sampling', SMOTE()),
        ("model", LogisticRegression(max_iter= 1000))
    ])
    my_score = {
        'Accuracy' : 'accuracy',
        'Recall' : 'recall',
        'Precision' : 'precision'
    }
    scores = cross_validate(model, X, Y, scoring = my_score, cv = 10)
    scores2 = cross_val_predict
    accuracy = sum(list(scores['test_Accuracy']))/10
    recall = sum(list(scores['test_Recall']))/10
    precision = sum(list(scores['test_Precision']))/10

    return accuracy, recall, precision

print(Evaluate(raw_data, feature))