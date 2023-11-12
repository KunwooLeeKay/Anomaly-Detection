
import Module_Preprocessing_New as prep

pp = prep.Preprocessing()

# 전처리
train_X, test_X, train_Y, test_Y = pp.B_Splitter()
train_X, test_X = pp.Normalize(train_X, test_X)

import Module_RandomForest_New as RFC
import xgboost as xgb
from xgboost import XGBClassifier


# model, param = RFC.RandomForest(train_X, train_Y)

model = XGBClassifier()

pred = model.predict(test_X)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_Y, pred))



# import Module_Evaluation as eval
# pred = model.predict(test_X)

# print(eval.Evaluation(test_Y, pred))
# eval.Plot_ROC(test_Y, model.predict_proba(test_X)[:,1], 'test')
