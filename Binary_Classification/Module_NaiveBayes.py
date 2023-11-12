
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
import numpy as np
from sklearn.model_selection import GridSearchCV

def NB(train_X, val_X, test_X, train_Y, val_Y, test_Y):

    # learn_model = GaussianNB() # 0.626
    # learn_model = BernoulliNB() # 0.743
    # learn_model = ComplementNB() # 0.652
    # learn_model = MultinomialNB() # 0.652

    learn_model = BernoulliNB() # 0.743
    
    train_X = np.concatenate((train_X, val_X), axis = 0)
    train_Y = np.concatenate((train_Y, val_Y), axis = 0)

    parameters ={'alpha' : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    scoring = ['accuracy', 'f1', 'roc_auc']

    gs = GridSearchCV(learn_model, parameters, scoring = scoring[2], cv = 10, n_jobs = -1)

    gs.fit(train_X, train_Y)

    best_model = gs.best_estimator_
    best_param = gs.best_params_

    return best_model, best_param

# import Module_Preprocessing as prep

# pp = prep.Preprocessing()

# X_binary, y_binary, X_multi, y_multi = pp.BorderlineSMOTE()
# train_X, val_X, test_X, train_Y, val_Y, test_Y = pp.B_Splitter(X_binary, y_binary)
# train_X, val_X, test_X, train_Y, val_Y, test_Y = pp.Normalize(train_X, val_X, test_X, train_Y, val_Y, test_Y)