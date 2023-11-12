from lib2to3.pgen2.literals import test
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import Module_Preprocessing_New as prep

pp = prep.Preprocessing()

train_X, test_X, train_Y, test_Y = pp.B_Splitter()

train_X, train_Y = pp.SMOTE(train_X, train_Y)

train_X, test_X = pp.Normalize(train_X, test_X)


# train_Y = train_Y.apply(lambda x : 1 if x == 0 else 0)
# test_Y = test_Y.apply(lambda x : 1 if x == 0 else 0)



learn_model = RandomForestClassifier(random_state=None, n_jobs = -1)

parameters = {'n_estimators' : range(90,111,10), 'criterion' : ['gini', 'entropy'] ,\
    'class_weight' : ['balanced', 'balanced_subsample'], 'max_features' : ['auto', 'sqrt', 'log2']}
parameters = {}

gs = GridSearchCV(learn_model, parameters, scoring = 'f1', cv = 10, n_jobs = -1)

gs.fit(train_X, train_Y)

best_model = gs.best_estimator_
best_param = gs.best_params_
import pandas as pd
df = pd.DataFrame(gs.cv_results_)
df.to_excel("Test.xlsx")

# import Module_Evaluation as Eval

# result = Eval.Evaluation(test_Y, gs.predict(test_X))
# print(result)


from sklearn.metrics import roc_curve
fpr, tpr, treshold = roc_curve(test_Y, best_model.predict_proba(test_X)[:,1])

from matplotlib import pyplot as plt

plt.plot(fpr, tpr)
plt.savefig(r"C:\Users\user\Desktop\ROC\figure1")