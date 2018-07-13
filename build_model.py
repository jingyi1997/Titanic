from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, RandomizedSearchCV, train_test_split
from sklearn import metrics
from scipy.stats import randint as sp_randint
from time import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, GradientBoostingClassifier

def build_model_gbc(X_train, y_train):
    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
    }
    kfold = StratifiedKFold(n_splits=10)
    gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv = kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
    gsGBC.fit(X_train,y_train)
    GBC_best = gsGBC.best_estimator_
    return GBC_best, gsGBC.best_score_

def build_model_rfc(X_train, y_train):
    RFC = RandomForestClassifier()


    ## Search grid for optimal parameters
    rf_param_grid = {"max_depth": [None],
                "max_features": [1, 3, 10],
                "min_samples_split": [2, 3, 10],
                "min_samples_leaf": [1, 3, 4, 10],
                "bootstrap": [False],
                "n_estimators" :[100,300],
                "criterion": ["gini"]}
    kfold = StratifiedKFold(n_splits=10)
    gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
    gsRFC.fit(X_train,y_train)

    RFC_best = gsRFC.best_estimator_
    return RFC_best, gsRFC.best_score_