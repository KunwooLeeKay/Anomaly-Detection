import seaborn as sns

import pandas as pd
raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")

feature = ['사용년수', '진동 횟수', '최대 기계압력', '최소 기계압력',\
     '기계 길이', '기계 무게', '특수 부품 저항', '일반 부품 저항', '기계 마력',\
          'O', 'P', '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J', 'L/J'] 
          # 연속형 데이터


from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
standard_scaler.fit(raw_data[feature])

raw_data[feature] = standard_scaler.transform(raw_data[feature])


import matplotlib.pyplot as plt

i = 0

for name in feature:
    sns.distplot(raw_data[name])
    plt.savefig(r'C:\Users\user\Desktop\정규분포/' + str(i) + '.png')
    plt.close()
    i += 1