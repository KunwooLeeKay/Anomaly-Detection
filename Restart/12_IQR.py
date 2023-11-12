import numpy as np
def detect_outliers(df, features):
    dic = {}
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_indices = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        dic[col] = outlier_indices
    return dic

import pandas as pd
raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")

feature = ['사용년수', '진동 횟수', '최대 기계압력', '최소 기계압력',\
     '기계 길이', '기계 무게', '특수 부품 저항', '일반 부품 저항', '기계 마력',\
          'O', 'P', '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J', 'L/J']

outliers = detect_outliers(raw_data, feature)
print(outliers)
outlier_rows = []
from collections import Counter

for col in feature:
    outlier_rows.extend(outliers[col])

count = dict(Counter(outlier_rows))
print(count)
print(len(list(count.keys())))

drop_row = list(count.keys())
drop_row.sort()

print(len(drop_row))

raw_data.drop(drop_row, axis = 0, inplace = True)


# def Evaluate(data, feature):

#     from sklearn.model_selection import cross_validate, cross_val_predict
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.linear_model import LogisticRegression
#     from imblearn.over_sampling import SMOTE
#     from imblearn.pipeline import Pipeline

#     X = data[feature]
#     Y = data['고장 유무']

#     standard_scaler = StandardScaler()
#     standard_scaler.fit(X)

#     X = standard_scaler.transform(X)

#     model = Pipeline([
#         ('sampling', SMOTE()),
#         ("model", LogisticRegression(max_iter= 1000))
#     ])
#     my_score = {
#         'Accuracy' : 'accuracy',
#         'Recall' : 'recall',
#         'Precision' : 'precision'
#     }
#     scores = cross_validate(model, X, Y, scoring = my_score, cv = 10)
#     scores2 = cross_val_predict
#     accuracy = sum(list(scores['test_Accuracy']))/10
#     recall = sum(list(scores['test_Recall']))/10
#     precision = sum(list(scores['test_Precision']))/10

#     return accuracy, recall, precision

# def Plot_Recall_Precision(data, feature, name):
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import precision_recall_curve
#     import numpy as np
#     from sklearn.model_selection import train_test_split
#     from imblearn.pipeline import Pipeline
#     from imblearn.over_sampling import SMOTE

#     X = data[feature]
#     Y = data['고장 유무']
        
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.linear_model import LogisticRegression

#     plt.figure(figsize=(8,6))

#     temp_precision, temp_recall = [], []

#     for random_seed in range(10):
#         train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state = random_seed, test_size = 0.2, stratify = Y)

#         standard_scaler = StandardScaler()
#         standard_scaler.fit(train_X)

#         train_X = standard_scaler.transform(train_X)
#         test_X = standard_scaler.transform(test_X)

#         model = Pipeline([
#             ('sampling', SMOTE()),
#             ('model', LogisticRegression(max_iter = 1000))
#         ])

#         learn_model = model.fit(train_X, train_Y)

#         pred_proba = learn_model.predict_proba(test_X)[:,1]
#         precisions, recalls, thresholds = precision_recall_curve(test_Y, pred_proba)
#         threshold_boundary = thresholds.shape[0]
        
#         temp_precision.append(precisions[0:threshold_boundary])
#         temp_recall.append(recalls[0:threshold_boundary])

#         plt.plot(thresholds, precisions[0:threshold_boundary],label = 'precision', linestyle = '--',color = 'b', aa = True, lw = 1.5)
#         plt.plot(thresholds, recalls[0:threshold_boundary], label = 'recall', color = 'r', aa = True, lw = 1.5)

#     plt.title('Recall - Precision')
#     start, end = plt.xlim()
#     plt.xticks(np.round(np.arange(start,end,0.1),2))
#     plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
#     plt.grid()
#     plt.show()

# print(Evaluate(raw_data, feature))
# Plot_Recall_Precision(raw_data, feature, 'Good')