from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV

def RandomForest(train_X, val_X, test_X, train_Y, val_Y, test_Y):

     # 자동으로 validation 나눠서 cross validation 하니까 다시 합친다음에 넣어주었다.
    train_X = np.concatenate((train_X, val_X), axis = 0)
    train_Y = np.concatenate((train_Y, val_Y), axis = 0)

    learn_model = RandomForestClassifier(random_state=None, n_jobs = -1)

    parameters = {'n_estimators' : range(90,111,2), 'criterion' : ['gini', 'entropy'] ,\
        'class_weight' : ['balanced', 'balanced_subsample'], 'max_features' : ['auto', 'sqrt', 'log2']}
    parameters = {}

    scoring = ['accuracy', 'f1', 'roc_auc']
    gs = GridSearchCV(learn_model, parameters, scoring = scoring[2], cv = 10, n_jobs = -1)

    gs.fit(train_X, train_Y)

    best_model = gs.best_estimator_
    best_param = gs.best_params_

    return best_model, best_param

def M_RandomForestClassifier(train_X, val_X, test_X, train_Y, val_Y, test_Y):

     # 자동으로 validation 나눠서 cross validation 하니까 다시 합친다음에 넣어주었다.
    train_X = np.concatenate((train_X, val_X), axis = 0)
    train_Y = np.concatenate((train_Y, val_Y), axis = 0)

    learn_model = RandomForestClassifier(random_state=None, n_jobs = -1)

    parameters = {'n_estimators' : range(80,121,5), 'class_weight' : ['balanced', 'balanced_subsample']}
    parameters = {}
    scoring = ['accuracy', 'f1', 'roc_auc']
    gs = GridSearchCV(learn_model, parameters, scoring = scoring[0], cv = 10, n_jobs = -1)

    gs.fit(train_X, train_Y)

    best_model = gs.best_estimator_
    return best_model 


# import Module_Preprocessing as prep

# pp = prep.Preprocessing()

# X_binary, y_binary, X_multi, y_multi = pp.BorderlineSMOTE()
# train_X, val_X, test_X, train_Y, val_Y, test_Y = pp.B_Splitter(X_binary, y_binary)
# train_X, val_X, test_X, train_Y, val_Y, test_Y = pp.Normalize(train_X, val_X, test_X, train_Y, val_Y, test_Y)

# model = RandomForestClassifier(train_X, val_X, test_X, train_Y, val_Y, test_Y)
# from sklearn.metrics import accuracy_score
# prediction = model.predict(test_X)
# print(accuracy_score(prediction, test_Y))

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# names = ['Accuracy','Precision','Recall', 'F1 score', 'AUC']
# scores = [accuracy_score(test_Y, prediction), precision_score(test_Y, prediction), \
#     recall_score(test_Y, prediction), f1_score(test_Y, prediction), roc_auc_score(test_Y, prediction)]

# from matplotlib import pyplot as plt

# plt.bar(names, scores, width = 0.4)
# plt.ylim(0.5, 1.0)
# plt.grid(True)
# plt.title("Scores")
# plt.show()
