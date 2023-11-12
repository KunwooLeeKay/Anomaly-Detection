from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def SVM(train_X, train_Y, sampler):

    
    parameters ={'C' : range(1,100,25), 'kernel' : ['rbf', 'linear', 'poly', 'sigmoid'],\
         'gamma' : ['scale', 'auto']}
    
    parameters = {}

    from imblearn.pipeline import Pipeline

    model = Pipeline([
            ('sampling', sampler),
            ('classifier', SVC(probability=True)) 
        ])

    grid = GridSearchCV(model, parameters, scoring='roc_auc')
    grid.fit(train_X, train_Y)

    best_model = grid.best_estimator_
    best_param = grid.best_params_


    return best_model, best_param
