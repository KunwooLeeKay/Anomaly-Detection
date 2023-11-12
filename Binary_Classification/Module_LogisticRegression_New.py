import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def LogisticRegressor(train_X, train_Y, sampler):
    
    from imblearn.pipeline import Pipeline


    parameters = {'classifier__C' : range(1,301,25),'classifier__penalty' : ['l2', 'none']}
    parameters = {}


    model = Pipeline([
            ('sampling', sampler),
            ('classifier', LogisticRegression(max_iter= 5000, n_jobs = -1)) 
        ])

    grid = GridSearchCV(model, parameters, scoring='f1', n_jobs = -1)
    grid.fit(train_X, train_Y)

    best_model = grid.best_estimator_
    best_param = grid.best_params_

    return best_model, best_param


def M_LogisticRegressor(train_X, val_X, test_X, train_Y, val_Y, test_Y):

    train_X = np.concatenate((train_X, val_X), axis = 0)
    train_Y = np.concatenate((train_Y, val_Y), axis = 0)

    learn_model = LogisticRegression(random_state= None, max_iter=5000, n_jobs=-1)


    parameters = {'C' : range(1,300,25)}
    scoring = ['accuracy', 'f1', 'roc_auc']

    gs = GridSearchCV(learn_model, parameters, scoring = scoring[0], cv = 10, n_jobs = -1)

    gs.fit(train_X, train_Y)

    best_model = gs.best_estimator_

    return best_model

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# def Evaluation(test_Y, prediction):
#     scores = {'Accuracy' : accuracy_score(test_Y, prediction),'Precision' : precision_score(test_Y, prediction),\
#         'Recall' : recall_score(test_Y, prediction), 'F1 score' : f1_score(test_Y, prediction),\
#              'AUC' : roc_auc_score(test_Y, prediction)}
#     return scores

# import Module_Preprocessing as prep

# pp = prep.Preprocessing()

# X_binary, y_binary, X_multi, y_multi = pp.BorderlineSMOTE()
# train_X, val_X, test_X, train_Y, val_Y, test_Y = pp.B_Splitter(X_binary, y_binary)
# train_X, val_X, test_X, train_Y, val_Y, test_Y = pp.Normalize(train_X, val_X, test_X, train_Y, val_Y, test_Y)

# lr, lr_param = LogisticRegressor(train_X, val_X, test_X, train_Y, val_Y, test_Y)

# lr_prediction = lr.predict(test_X)

# # nb_prediction = nb.predict(test_X)

# predictions = [lr_prediction]
# param = [lr_param]
# model_list = ['LR']

# oversampling = 'SMOTE'
# B_classification = {}

# for num in range(0,len(predictions)):
#     param = tuple(param[num].items())
#     method = (oversampling, model_list[num], param)

#     print(method)
#     B_classification[method] = predictions[num]
#     B_classification[method] = Evaluation(test_Y, predictions[num])

# print(B_classification)