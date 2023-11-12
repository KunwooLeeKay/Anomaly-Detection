import pandas as pd
raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")


feature = ['사용년수', '진동 횟수', '최대 기계압력', '최소 기계압력',\
     '기계 길이', '기계 무게', '특수 부품 저항', '일반 부품 저항', '기계 마력',\
          'O', 'P', '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J', 'L/J'] 
          # 연속형 데이터

corr_matrix = raw_data[feature].corr()


from itertools import combinations
combi = list(combinations(feature, 2))

no_relation = []
bar = 0.8 # BAR 를 조정하면 공분산을 몇 이상에서 자를지 결정해준다

for i in range(0, len(combi)):
    correlation = abs(corr_matrix[combi[i][0]][combi[i][1]])
    if correlation < bar:
        combination = list(combi[i])
        temp_feature = ['사용년수', '진동 횟수', '최대 기계압력', '최소 기계압력',\
        '기계 길이', '기계 무게', '특수 부품 저항', '일반 부품 저항', '기계 마력',\
            'O', 'P', '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J', 'L/J']
        temp_combination = combination
        for name in combination:
            temp_feature.remove(name)
            for element in temp_feature:
                flag = True
                for j in range(0, len(temp_combination)):
                    if abs(corr_matrix[combination[j]][element]) >= bar:
                        flag = False
                if flag == True:
                    temp_combination.append(element)

        no_relation.append(temp_combination)

print(no_relation)
print(len(no_relation))

import pickle

with open('feature.pickle', 'wb') as pickle_data:
    pickle.dump(no_relation, pickle_data)