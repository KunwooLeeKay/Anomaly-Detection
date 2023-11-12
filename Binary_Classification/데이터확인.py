import Module_Preprocessing_New as prep
from matplotlib import pyplot as plt

pp = prep.Preprocessing()


# for splitter in ['B_Splitter']:
#     for oversampling in ['SMOTE', 'BorderlineSMOTE', 'ADASYN']:
#         for scaler in ['Normalize', 'Standardize']:
#             print(splitter, oversampling, scaler)
#             # 전처리
#             train_X, test_X, train_Y, test_Y = prep.Preprocessing.__dict__[splitter](pp)
#             train_X, train_Y = prep.Preprocessing.__dict__[oversampling](pp, train_X, train_Y)
#             train_X, test_X = prep.Preprocessing.__dict__[scaler](pp,train_X, test_X)
#             plt.plot()


data = pp.raw_data

feature = pp.feature

print(data[feature[2]])