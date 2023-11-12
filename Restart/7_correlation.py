
from timeit import repeat
import pandas as pd
raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")
whole_feature = ['제조국가', '사용년수', 'C', 'D', 'E', '진동 횟수', '최대 기계압력', '최소 기계압력', '기계 길이',\
                        '기계 무게', '특수 부품 유무', '특수 부품 저항', '일반 부품 저항', '기계 마력', 'O', 'P',\
                        '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J',\
                        'L/J']

feature = ['제조국가', 'C', 'D', 'E', '특수 부품 유무'] # 범주형 데이터

for name in feature:
    whole_feature.remove(name)

feature = whole_feature # 수치형 데이터
# print(len(feature))

corr_matrix = raw_data[feature].corr()

from itertools import combinations
combi = list(combinations(feature, 2))
print(combi)

# print(corr_matrix[combi[3][0]][combi[3][1]])

no_relation = []

related_values_5 = []
related_values_6 = []
related_values_7 = []
related_values_8 = []
related_values_highly = []

total = {}

for i in range(0, len(combi)):

    correlation = abs(corr_matrix[combi[i][0]][combi[i][1]])
    combination = tuple((combi[i][0], combi[i][1]))
    total[combination] = {'correlation' : round(correlation, 3)}

    if correlation < 0.5:
        no_relation.append(combination)

    if 0.5 <= correlation < 0.6:
        # print("{} 와 {} 는 0.5 상관관계가 있습니다.".format(combi[i][0], combi[i][1]))
        related_values_5.append(combination)

    elif 0.6 <= correlation < 0.7:

        related_values_6.append(combination)

    elif 0.7 <= correlation < 0.8:
        related_values_7.append(combination)

    elif 0.7 <= correlation < 0.8:

        related_values_8.append(combination)

    else:
        related_values_highly.append(combination)

    

print(len(related_values_5), len(related_values_6), len(related_values_7), len(related_values_8), len(related_values_highly))
# print(related_values_highly)
# print(no_relation)

print(total)

df = pd.DataFrame(total)
df.to_excel('correlations.xlsx')

for i in range():
    correlation = abs(corr_matrix[combi[i][0]][combi[i][1]])
    combination = tuple((combi[i][0], combi[i][1]))