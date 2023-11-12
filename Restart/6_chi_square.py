
import pandas as pd
raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")
feature = ['제조국가', '사용년수', 'C', 'D', 'E', '진동 횟수', '최대 기계압력', '최소 기계압력', '기계 길이',\
                        '기계 무게', '특수 부품 유무', '특수 부품 저항', '일반 부품 저항', '기계 마력', 'O', 'P',\
                        '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J',\
                        'L/J']

feature = ['제조국가', 'C', 'D', 'E', '특수 부품 유무'] # 범주형 데이터
target = ['고장 단계', '고장 유무']

f1 = raw_data['제조국가']
f2 = raw_data['C']

ctab = pd.crosstab(f1, f2)
print(ctab)

from scipy.stats import chi2_contingency

# chival, pval, df, exp = chi2_contingency(ctab)
# print(chival, pval, df, exp, sep = '\n') 
# 이때 pval 이 0.05 보다 작으면 둘이 연관이 있다고 보면 된다.
# 추가로 exp에서 5이하의 값이 20% 이상을 차지한다면 카이제곱 검정을 사용하면 안된다.
# 만약 기대 빈도가 높다면 정규분포를 근사할 수 있다.



# df = raw_data[feature]
# print(df.describe())
# for column in feature:
#     print(raw_data[column].value_counts())


# 제조국가 : 1, 2
# C : 1, 2
# D : 1, 2, 3
# E : 8, 1, 2
# 특수 부품 유무 : 1, 9 
temp_list = []

from itertools import combinations
combi = list(combinations(feature, 2))

for i in range(0, len(combi)):
    column1 = combi[i][0]
    column2 = combi[i][1]
    ctab = pd.crosstab(raw_data[column1], raw_data[column2])
    chival, pval, df, exp = chi2_contingency(ctab)
    if len(exp[exp < 5])/len(exp)*100 >= 20:
        print("{}와 {}는 카이제곱 검정을 사용할 수 없습니다.".format(column1, column2))
    else:
        if pval < 0.05:
            print("{}와 {}는 종속이다.".format(column1, column2))
        else:
            print("{}와 {}는 독립!!!".format(column1, column2))


column1 = combi[0][0]
column2 = combi[0][1]
ctab = pd.crosstab(raw_data[column1], raw_data[column2])
print(ctab)
chival, pval, df, exp = chi2_contingency(ctab)
print(chival)
print(df)
print(exp)
print(pval)

if len(exp[exp < 5])/len(exp)*100 >= 20:
    print("{}와 {}는 카이제곱 검정을 사용할 수 없습니다.".format(column1, column2))
else:
    if pval < 0.05:
        print("{}와 {}는 종속이다.".format(column1, column2))
    else:
        print("{}와 {}는 독립!!!".format(column1, column2))

# 분석결과

# 결론 : 
# D가 들어가면 카이제곱 검정을 사용할 수 없음. - > D를 사용하여 피셔제곱검정을 이용하자

# 제조국가 - C, E 는 종속
# 제조국가 - 특수 부품 유무 독립
# C - 특수부품 유무 독립
# C - E 독립
# E - 특수부품 유무는 독립

# C - E - 특수부품 유무 는 상호 독립

# 그러니까 C, E, 특수부품유무에다가 D를 넣고 빼서 시도해보자!



# 얘는 피셔의정확검정인데 2X2 만 되서 D에다가 못써먹음. 원래는 카이제곱 안되면(exp<5 20%) 이거 해봐야함.
# import scipy.stats as stats

# for column in feature:
#     if column == 'D':
#         continue
#     ctab = pd.crosstab(raw_data['D'],raw_data[column])
#     oddratio, pval = stats.fisher_exact(ctab)
#     if pval < 0.05:
#         print("{}와 {}는 종속이다.".format(column1, column2))
#     else:
#         print("{}와 {}는 독립!!!".format(column1, column2))