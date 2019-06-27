import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics.scorer import make_scorer

#read data
train = pd.read_csv("../Data/train/train.csv")
test = pd.read_csv("../Data/test/test.csv")
colors = pd.read_csv("../Data/color_labels.csv")
breeds = pd.read_csv("../Data/breed_labels.csv")

#Preprocess data
train=preprocess(train.copy())
test=preprocess(test.copy())

##Train random forest classifier
# Random Hyperparameter Grid
random_grid={
    'bootstrap': [False,True],
    'max_depth': [10,20,30,40,50,60,70,80,90,100,None],
    'max_features': ['auto','sqrt'],
    'min_samples_leaf': [1,2,4],
    'min_samples_split': [2,5,10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}

my_scorer = make_scorer(quadratic_weighted_kappa, greater_is_better=True)
rf=RandomForestClassifier()
#For random search use n_iter 600 and cv 5, Then use grid search with cv = 10
rf_random=RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 600, cv = 5, verbose = 0, random_state=11,n_jobs=4,scoring=my_scorer)
rf_random.fit(train,labels)
rf_random.best_params_


rf=RandomForestClassifier(n_estimators= 400,min_samples_split= 2,min_samples_leaf= 2,max_features= 'sqrt',max_depth= 30,bootstrap= False,verbose=0,random_state=11)
rf.fit(train,labels)
predictions=rf.predict(test)
#CV = 0.379, LB 0.313

np.unique(predictions, return_counts=True)
feature_importances = pd.DataFrame(rf.feature_importances_,index = train.columns,columns=['importance']).sort_values('importance',ascending=False)
