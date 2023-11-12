from sklearn.metrics import confusion_matrix
import pickle

with open("Model.pickle",'rb') as pickle_data:
    learn_model = pickle.load(pickle_data)

print(learn_model)

learn_model = learn_model[('ADASYN', 'Normalize', 'B_Splitter')][1]
print(learn_model)

import Module_Preprocessing_New as prep
pp = prep.Preprocessing()

for splitter in ['B_Splitter']:
    for oversampling in ['SMOTE', 'BorderlineSMOTE', 'ADASYN']:
        for scaler in ['Normalize', 'Standardize']:
            print(splitter, oversampling, scaler)
            # 전처리
            train_X, test_X, train_Y, test_Y = prep.Preprocessing.__dict__[splitter](pp)
            
            train_X, train_Y = prep.Preprocessing.__dict__[oversampling](pp, train_X, train_Y)
            # test_X, test_Y = prep.Preprocessing.__dict__[oversampling](pp, test_X, test_Y)
            train_X, test_X = prep.Preprocessing.__dict__[scaler](pp,train_X, test_X)

learn_model.fit(train_X, train_Y)
pred = learn_model.predict(test_X)
print(confusion_matrix(test_Y, pred))

test_Y = test_Y.tolist()
pred = pred.tolist()

dic = {'real' : test_Y, 'pred' : pred}
import pandas as pd
df = pd.DataFrame(dic)
df.to_excel("tt.xlsx")

# import numpy as np
# print(type(test_Y))
# print(type(pred))

# l = np.concatenate(test_Y, pred, axis = 1)

# import pandas as pd

# df = pd.DataFrame(l,columns=['REAL', 'PREDICTION'])
# print(df)