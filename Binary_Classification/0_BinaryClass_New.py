import Module_Preprocessing_New as prep
import Module_Preprocessing_New as prep
import Module_RandomForest_New as RF
import Module_LogisticRegression_New as LR
import Module_SVM_New as SVM
import Module_Evaluation as Eval
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

pp = prep.Preprocessing()

B_classification = {}
Best_Model = {}

for splitter in ['B_Splitter']:
    for oversampling in ['SMOTE', 'BorderlineSMOTE', 'ADASYN']:

      if oversampling == 'SMOTE' : 
        sampler = SMOTE()
      elif oversampling == 'BorderlineSMOTE' : 
        sampler = BorderlineSMOTE()
      elif oversampling == 'ADASYN' : 
        sampler = ADASYN()

        for scaler in ['Normalize', 'Standardize']:
            print(splitter, oversampling, scaler)
            # 전처리
            train_X, test_X, train_Y, test_Y = prep.Preprocessing.__dict__[splitter](pp)
            # test_X, test_Y = prep.Preprocessing.__dict__[oversampling](pp, test_X, test_Y)
            train_X, test_X = prep.Preprocessing.__dict__[scaler](pp,train_X, test_X)

            # 학습
            rf, rf_param = RF.RandomForest(train_X,train_Y, sampler)
            lr, lr_param = LR.LogisticRegressor(train_X,train_Y, sampler)
            svm, svm_param = SVM.SVM(train_X,train_Y, sampler)

            # 예측
            rf_prediction = rf.predict(test_X)
            lr_prediction = lr.predict(test_X)
            svm_prediction = svm.predict(test_X)

            predictions = [rf_prediction, lr_prediction, svm_prediction]
            param = [rf_param, lr_param, svm_param]
            model_list = ['RFC', 'LR', 'SVM']
            learn_model = [rf, lr, svm]

            for num in range(0,len(predictions)):
                tup_param = tuple(param[num].items())
                method = (oversampling, scaler, model_list[num], tup_param)
                B_classification[method] = Eval.Evaluation(test_Y, predictions[num])
                roc_title = (model_list[num], oversampling, scaler)
                Eval.Plot_ROC(test_Y, learn_model[num].predict_proba(test_X)[:,1], roc_title)
                Eval.Plot_Recall_Precision(test_Y, learn_model[num].predict_proba(test_X)[:,1], roc_title)

            method = (oversampling, scaler, splitter)
            learn_model = [rf, lr, svm]
            Best_Model[method] = learn_model

            
Eval.Save(B_classification)


import pickle
import os
 
filename='BC' #파일명 고정값
file_ext='.pickle' #파일 형식

uniq=1
output_path='C:/Users/user/Desktop/LearnPython/%s%s' % (filename,file_ext)

while os.path.exists(output_path):  #동일한 파일명이 존재할 때
  output_path='C:/Users/user/Desktop/LearnPython/%s(%d)%s' % (filename,uniq,file_ext) #파일명(1) 파일명(2)...
  uniq+=1

with open(output_path, "wb") as pickle_data:
    pickle.dump(B_classification, pickle_data)


 
filename='Model' #파일명 고정값
file_ext='.pickle' #파일 형식

uniq=1
output_path='C:/Users/user/Desktop/LearnPython/%s%s' % (filename,file_ext)

while os.path.exists(output_path):  #동일한 파일명이 존재할 때
  output_path='C:/Users/user/Desktop/LearnPython/%s(%d)%s' % (filename,uniq,file_ext) #파일명(1) 파일명(2)...
  uniq+=1

with open(output_path, "wb") as pickle_data:
    pickle.dump(Best_Model, pickle_data)

