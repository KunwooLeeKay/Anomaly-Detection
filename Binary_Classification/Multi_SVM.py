import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def SVM(train_X, train_Y):

    learn_model = SVC(random_state = None, n_jobs = -1)


    parameters ={'C' : range(1,100,25), 'kernel' : ['rbf', 'linear', 'poly', 'sigmoid'],\
         'gamma' : ['scale', 'auto']}
    
    parameters = {}

    scoring = ['accuracy', 'f1', 'roc_auc']

    gs = GridSearchCV(learn_model, parameters, scoring = scoring[2], cv = 10, n_jobs = -1)

    gs.fit(train_X, train_Y)

    best_model = gs.best_estimator_
    best_param = gs.best_params_

    return best_model, best_param

def M_SVM(train_X, val_X, test_X, train_Y, val_Y, test_Y):

    train_X = np.concatenate((train_X, val_X), axis = 0)
    train_Y = np.concatenate((train_Y, val_Y), axis = 0)

    learn_model = SVC(random_state = None)

    parameters ={'C' : range(1,100,25),'decision_function_shape':['ovo'],\
         'kernel' : ['rbf', 'linear'], 'gamma' : ['scale', 'auto'],}
    scoring = ['accuracy', 'f1', 'roc_auc']

    gs = GridSearchCV(learn_model, parameters, scoring = scoring[0], cv = 10, n_jobs = -1)

    # GridSearchCV 할때 scoring이 roc_auc면 multiclass 분류가 안됨!!! 

    gs.fit(train_X, train_Y)

    best_model = gs.best_estimator_

    return best_model

# import Module_Preprocessing as prep

# pp = prep.Preprocessing()

# X_binary, y_binary, X_multi, y_multi = pp.BorderlineSMOTE()
# train_X, val_X, test_X, train_Y, val_Y, test_Y = pp.B_Splitter(X_multi, y_multi)
# train_X, val_X, test_X, train_Y, val_Y, test_Y = pp.Normalize(train_X, val_X, test_X, train_Y, val_Y, test_Y)

# import numpy as np
# train_X = np.concatenate((train_X, val_X), axis = 0)
# train_Y = np.concatenate((train_Y, val_Y), axis = 0)

# from sklearn.svm import SVC

# learn_model = SVC(random_state=1)

# from sklearn.model_selection import GridSearchCV

# parameters ={'C' : range(1,100,25),'decision_function_shape':['ovo', 'ovr'],\
#         'kernel' : ['rbf', 'linear'], 'gamma' : ['scale', 'auto']}
# scoring = ['accuracy', 'f1', 'roc_auc']

# gs = GridSearchCV(learn_model, parameters, scoring = scoring[0], cv = 10, n_jobs = -1)

# gs.fit(train_X, train_Y)

# svm = gs.best_estimator_



# # svm = SVM(train_X, val_X, test_X, train_Y, val_Y, test_Y)
# from sklearn.metrics import accuracy_score
# prediction = svm.predict(test_X)
# print(accuracy_score(prediction, test_Y))
