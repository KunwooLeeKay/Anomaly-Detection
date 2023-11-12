from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def RandomForest(train_X, train_Y, sampler):

    parameters = {'classifier__n_estimators' : range(90,111,5), 'classifier__min_samples_split' : range(2,11),\
         'classifier__max_features' : [None, 'sqrt']}
    parameters = {}
    
    from imblearn.pipeline import Pipeline

    model = Pipeline([
            ('sampling', sampler),
            ('classifier', RandomForestClassifier()) 
        ])

    print(RandomForestClassifier().get_params().keys())

    grid = GridSearchCV(model, parameters, scoring='roc_auc')
    grid.fit(train_X, train_Y)

    best_model = grid.best_estimator_
    best_param = grid.best_params_

    return best_model, best_param


