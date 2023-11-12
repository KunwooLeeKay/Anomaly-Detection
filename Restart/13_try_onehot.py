
import pandas as pd
raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")
# raw_data.drop([154,236,207, 701, 588, 601, 3, 246, 934, 441, 615, 458], axis = 0, inplace = True)
feature = ['제조국가', '사용년수', 'C', 'D', 'E', '진동 횟수', '최대 기계압력', '최소 기계압력', '기계 길이',\
                        '기계 무게', '특수 부품 유무', '특수 부품 저항', '일반 부품 저항', '기계 마력', 'O', 'P',\
                        '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J',\
                        'L/J']

# 원 핫 인코딩
target = ['제조국가', 'C', 'D', 'E', '특수 부품 유무']

encoded_data = pd.get_dummies(raw_data, columns = target)
print(encoded_data.columns)
encoded_feature = ['사용년수', '진동 횟수', '최대 기계압력', '최소 기계압력', '기계 길이', '기계 무게', '특수 부품 저항',
       '일반 부품 저항', '기계 마력', 'O', 'P', '기계 부품 임피던스', 'P/J', 'S', 'T', 'U',
       '기계 평균 압력', 'R/J', 'M*J', 'L*J', 'L/J', '제조국가_1',
       '제조국가_2', 'C_1', 'C_2', 'D_1', 'D_2', 'D_3', 'E_1', 'E_2', 'E_8',
       '특수 부품 유무_1', '특수 부품 유무_9']

from sklearn.model_selection import train_test_split
X = encoded_data[encoded_feature]
Y = encoded_data['고장 유무']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state = None, test_size = 0.2, stratify = Y)

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
standard_scaler.fit(train_X)

train_X = standard_scaler.transform(train_X)
test_X = standard_scaler.transform(test_X)


from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score

ftwo_scorer = make_scorer(fbeta_score, beta = 2)

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

model = Pipeline([
    ('sampling', SMOTE()),
    ("model", LogisticRegression(max_iter= 1000))
])

learn_model = model.fit(train_X, train_Y)
pred = learn_model.predict(test_X)
pred2 = learn_model.predict(train_X)
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
recall, precision, accuracy = recall_score(test_Y, pred), precision_score(test_Y, pred), accuracy_score(test_Y, pred)
# recall, precision, accuracy = recall_score(train_Y, pred2), precision_score(train_Y, pred2), accuracy_score(train_Y, pred2)

print(confusion_matrix(test_Y, pred))
print("recall : {}, precision : {}, accuracy : {}".format(recall, precision, accuracy))


def Plot_ROC(test_Y, proba, method):
    # ROC 커브를 그려줌
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(test_Y, proba)
    from matplotlib import pyplot as plt
    print(method)

    plt.plot(fpr, tpr, label = 'ROC')
    plt.plot([0,1], [0,1], 'k--', label = '50%')
    plt.show()
    # plt.savefig("C:/Users/user/Desktop/ROC/" + str(method))
    # plt.close()

def Plot_Recall_Precision(test_Y, pred_proba, method):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    precisions, recalls, thresholds = precision_recall_curve(test_Y, pred_proba)
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle = '--', label = 'precision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label = 'recall')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()
    # plt.savefig("C:/Users/user/Desktop/Precision_Recall/" + str(method))
    # plt.close()


Plot_ROC(test_Y, learn_model.predict_proba(test_X)[:,1], 'ROC')
Plot_Recall_Precision(test_Y, learn_model.predict_proba(test_X)[:,1], 'Recall_Precision')