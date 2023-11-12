import pandas as pd
raw_data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")
# feature = ['제조국가', '사용년수', 'C', 'D', 'E', '진동 횟수', '최대 기계압력', '최소 기계압력', '기계 길이',\
#                         '기계 무게', '특수 부품 유무', '특수 부품 저항', '일반 부품 저항', '기계 마력', 'O', 'P',\
#                         '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J',\
#                         'L/J']

qualitative = ['제조국가', 'C', 'D', 'E', '특수 부품 유무'] # 범주형 데이터

quantitative = ['사용년수', '진동 횟수', '최대 기계압력', '최소 기계압력',\
     '기계 길이', '기계 무게', '특수 부품 저항', '일반 부품 저항', '기계 마력',\
          'O', 'P', '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J', 'L/J'] 

target = ['고장 유무']


from scipy.stats import chi2_contingency

for name in qualitative:
    ctab = pd.crosstab(raw_data[name], raw_data['고장 유무'])
    chival, pval, df, exp = chi2_contingency(ctab)
    if len(exp[exp < 5])/len(exp)*100 >= 20:
        print("{}와 {}는 카이제곱 검정을 사용할 수 없습니다.".format(name, '고장 유무'))
    else:
        if pval < 0.05:
            print("{}와 {}는 종속이다.".format(name, '고장 유무'))
        else:
            print("{}와 {}는 독립!!!".format(name, '고장 유무'))
