import pickle

with open("Select_Algorithm.pickle", 'rb') as pickle_data:
    data = pickle.load(pickle_data)

with open("BC(1).pickle", 'rb') as pickle_data:
    data2 = pickle.load(pickle_data)
print(data2)
print("\n\n\n")
print(data)

import pandas as pd

df_list = []

new_dic = {}

model_names = ['LogisticRegression', 'RidgeClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'SVC', 'MLPClassifier', 'GaussianProcessClassifier','RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']
scaler_names = ['Normalize', 'Standardize']
sampler_names = ['SMOTE', 'BorderlineSMOTE', 'SVMSMOTE', 'ADASYN', 'SMOTETomek', 'SMOTEENN']
pass_score = {}
superior_score = {}

for model in model_names:

    for scaler in scaler_names:

        for sampler in sampler_names:

            temp_tup = (model, scaler, sampler)
            new_dic[temp_tup] = {'F2 Score' : data[model][scaler][sampler]}
            if float(data[model][scaler][sampler]) > 0.35:
                pass_score[model, scaler, sampler] = {'F2 Score' : data[model][scaler][sampler]}
                if float(data[model][scaler][sampler]) > 0.4:
                    superior_score[model, scaler, sampler] = {'F2 Score' : data[model][scaler][sampler]}

df = pd.DataFrame(new_dic)
df.to_excel("TryDifferentAlgorithms.xlsx")

print(pass_score)
print(superior_score)

df2 = pd.DataFrame(pass_score)
df3 = pd.DataFrame(superior_score)

df2.to_excel("PassedAlgorithms.xlsx")
df3.to_excel("SuperiorAlgoriths.xlsx")