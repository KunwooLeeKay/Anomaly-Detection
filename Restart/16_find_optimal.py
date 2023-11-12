from distutils import core
import pandas as pd
from sklearn.model_selection import cross_val_predict
raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")
feature = ['제조국가', '사용년수', 'C', 'D', 'E', '진동 횟수', '최대 기계압력', '최소 기계압력', '기계 길이',\
                        '기계 무게', '특수 부품 유무', '특수 부품 저항', '일반 부품 저항', '기계 마력', 'O', 'P',\
                        '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J',\
                        'L/J']

qualitative = ['제조국가', 'C', 'D', 'E', '특수 부품 유무'] # 범주형 데이터

quantitative = ['사용년수', '진동 횟수', '최대 기계압력', '최소 기계압력',\
     '기계 길이', '기계 무게', '특수 부품 저항', '일반 부품 저항', '기계 마력',\
          'O', 'P', '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J', 'L/J'] 


############ 카이제곱 검정 ############

# 제조국가 - C, E 는 종속
# 제조국가 - 특수 부품 유무 독립
# C - 특수부품 유무 독립
# C - E 독립
# E - 특수부품 유무는 독립

# C - E - 특수부품 유무 는 상호 독립

# 그러니까 C, E, 특수부품유무에다가 D를 넣고 빼서 시도해보자! -> D는 넣는게 더 좋은듯

from scipy.stats import chi2_contingency

from itertools import combinations
combi = list(combinations(qualitative, 2)) # 범주형 데이터 짝 생성
print(combi)

for i in range(0, len(combi)):
    column1 = combi[i][0]
    column2 = combi[i][1]
    ctab = pd.crosstab(raw_data[column1], raw_data[column2])
    chival, pval, df, exp = chi2_contingency(ctab)
    if len(exp[exp < 5])/len(exp)*100 >= 20:
        # print("{}와 {}는 카이제곱 검정을 사용할 수 없습니다.".format(column1, column2))
        pass
    else:
        if pval < 0.05:
            print("{}와 {}는 종속이다.".format(column1, column2))
        else:
            print("{}와 {}는 독립!!!".format(column1, column2))


independent_qualitative = [['C', 'E', '특수 부품 유무'] , ['제조국가', '특수 부품 유무']] # 상호 독립인 피쳐. 
unknown_qualitative = ['D'] # D 는 독립성을 검정할 수 없다.

qual_sets = [qualitative] # 확인해봐야할 범주형 데이터 세트 리스트
for i in range(0, len(independent_qualitative)):
    qual_sets.append(independent_qualitative[i])
    qual_sets.append(independent_qualitative[i] + unknown_qualitative)


############ 공분산 확인 ############
def Correlation(bar):
    corr_matrix = raw_data[quantitative].corr()
    from itertools import combinations
    combi = list(combinations(quantitative, 2))

    no_relation = []
    # BAR 를 조정하면 공분산을 몇 이상에서 자를지 결정해준다

    for i in range(0, len(combi)):
        correlation = abs(corr_matrix[combi[i][0]][combi[i][1]])
        if correlation < bar:
            combination = list(combi[i])
            print("initial combi : ", combination)
            temp_feature = ['사용년수', '진동 횟수', '최대 기계압력', '최소 기계압력',\
            '기계 길이', '기계 무게', '특수 부품 저항', '일반 부품 저항', '기계 마력',\
                'O', 'P', '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J', 'L/J']

            temp_combination = combination
            for name in combination:
                temp_feature.remove(name)
            print("temp feature : ", temp_feature) # 선택한 조합을 제외한 나머지 피처값 리스트

            for element in temp_feature:
                flag = True
                for j in range(0, len(temp_combination)):
                    if abs(corr_matrix[combination[j]][element]) >= bar:
                        flag = False
                        # time.sleep(1)
                if flag== False:
                    print("FAILED : ", element)
                if flag == True:
                    temp_combination.append(element)
                    print("###SUCCESS!! : ", element)
            
            no_relation.append(temp_combination)
    
    new = []
    for i in range(0, len(no_relation)):
        temp_list = no_relation[i]
        temp_list.sort()
        new.append(temp_list)

    final = []
    for i in range(0, len(new)):
        temp = list(new)
        temp.remove(temp[i])
        if new[i] not in temp:
            final.append(new[i])

    return final


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
    accuracy = sum(list(scores['test_Accuracy']))/10
    recall = sum(list(scores['test_Recall']))/10
    precision = sum(list(scores['test_Precision']))/10

    return accuracy, recall, precision

def Plot_Recall_Precision(data, feature, name):
    from sklearn.metrics import precision_recall_curve
    import numpy as np
    from sklearn.model_selection import train_test_split
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE

    X = data[feature]
    Y = data['고장 유무']
        
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression



    temp_precision, temp_recall = [], []

    for random_seed in range(10):
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state = random_seed, test_size = 0.2, stratify = Y)

        standard_scaler = StandardScaler()
        standard_scaler.fit(train_X)

        train_X = standard_scaler.transform(train_X)
        test_X = standard_scaler.transform(test_X)

        model = Pipeline([
            ('sampling', SMOTE()),
            ('model', LogisticRegression(max_iter = 1000))
        ])

        learn_model = model.fit(train_X, train_Y)

        pred_proba = learn_model.predict_proba(test_X)[:,1]
        precisions, recalls, thresholds = precision_recall_curve(test_Y, pred_proba)
        threshold_boundary = thresholds.shape[0]
        
        temp_precision.append(precisions[0:threshold_boundary])
        temp_recall.append(recalls[0:threshold_boundary])

        plt.plot(thresholds, precisions[0:threshold_boundary],label = 'precision', linestyle = '--',color = 'b', aa = True, lw = 1.5)
        plt.plot(thresholds, recalls[0:threshold_boundary], label = 'recall', color = 'r', aa = True, lw = 1.5)

    plt.title('Recall - Precision')
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.grid()
    plt.savefig("C:/Users/user/Desktop/LearnPython/고장진단프로젝트/Restart/Logistic_Regression_Recall_Precision/Recall_Precision" + str(name))
    plt.clf()

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))

whole_data = {}
file_no = 1
for bar in [0.4, 0.5, 0.6, 0.7, 0.8]:
    print(bar)
    quant_sets = Correlation(bar)

    for qual_set in qual_sets:
        print(qual_set)
        for quant_set in quant_sets:

            raw_feature = qual_set + quant_set

            encoded_data = pd.get_dummies(raw_data, columns = qual_set)

            onehot_feature = list(encoded_data.columns)
            onehot_feature.remove('고장 유무')
            onehot_feature.remove('고장 단계')
            encoded_accuracy, encoded_recall, encoded_precision = Evaluate(encoded_data, onehot_feature)
            accuracy, recall, precision = Evaluate(raw_data, raw_feature)
            whole_data[tuple((bar, tuple(raw_feature), 'One-Hot'))] = {'accuracy' : encoded_accuracy, 'recall' : encoded_recall, 'precision' : encoded_precision}
            whole_data[tuple((bar, tuple(raw_feature), 'No Encoding'))] = {'accuracy' : accuracy, 'recall' : recall, 'precision' : precision}
            Plot_Recall_Precision(encoded_data, onehot_feature, "One-Hot" + str(file_no))
            Plot_Recall_Precision(raw_data, raw_feature, "No Encoding" + str(file_no))
            file_no += 1

import pickle
with open('find_optimal.pickle', 'wb') as pickle_data:
    pickle.dump(whole_data, pickle_data)

### 이상치 제거 시도

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

# 고장 유무와 종속인 범주형 변수는 C, D - 카이제곱 검정
# 고장 유무와 종속인 연속형 변수는 .. 로지스틱 회귀 분석이나 판별 분석을 해야하는데 일단 C, D 에서 나온걸로 이상치 제거해보면 결과 팍 떨어짐... 굳이 해봐야할까?

from collections import Counter

whole_data = {}
file_no = 1
for bar in [0.4, 0.5, 0.6, 0.7, 0.8]:
    print(bar)
    quant_sets = Correlation(bar)

    for qual_set in qual_sets:
        print(qual_set)
        for quant_set in quant_sets:

            cautious_feature = quant_set
            outliers = detect_outliers(raw_data, cautious_feature)

            outlier_rows = []

            for col in cautious_feature:
                outlier_rows.extend(outliers[col])

            count = dict(Counter(outlier_rows))
    
            drop_row = list(count.keys())
            drop_row.sort()

            iqr_data = raw_data.drop(drop_row, axis = 0, inplace = False) # 이상치를 제거한 새로운 데이터셋 생성

            raw_feature = qual_set + quant_set

            encoded_data = pd.get_dummies(iqr_data, columns = qual_set)

            onehot_feature = list(encoded_data.columns)
            onehot_feature.remove('고장 유무')
            onehot_feature.remove('고장 단계')

            encoded_accuracy, encoded_recall, encoded_precision = Evaluate(encoded_data, onehot_feature)
            accuracy, recall, precision = Evaluate(iqr_data, raw_feature)


            whole_data[tuple((bar, tuple(raw_feature), 'One-Hot'))] = {'accuracy' : encoded_accuracy, 'recall' : encoded_recall, 'precision' : encoded_precision}
            whole_data[tuple((bar, tuple(raw_feature), 'No Encoding'))] = {'accuracy' : accuracy, 'recall' : recall, 'precision' : precision}
            Plot_Recall_Precision(encoded_data, onehot_feature, "IQR_One-Hot" +str(file_no))
            Plot_Recall_Precision(raw_data, raw_feature, "IQR_No Encoding" + str(file_no))
            file_no += 1

import pickle
with open('IQR_find_optimal.pickle', 'wb') as pickle_data:
    pickle.dump(whole_data, pickle_data)
