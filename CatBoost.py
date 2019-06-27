import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer

import scipy as sp
from preprocess import preprocess
import QuadraticKappa as qk
from functools import partial

from sklearn.model_selection import StratifiedKFold

#Preprocess data
# train=preprocess('train')
# test=preprocess('test')
# train.to_csv('../Data/train/trainProc.csv',index=False)
# test.to_csv('../Data/test/testProc.csv',index=False)

#Loas Preprocessed data
train_full = pd.read_csv('../Data/train/trainProc_fineTune.csv')
#We drop 'RescuerID' as there is not overlap between train and test
#We drop 'PhotoAmt' as we already have 'PhotoAmtGood' and 'PhotoAmtFrac=PhotoAmtGood/PhotoAmt'
toDrop = ['PetID','RescuerID','PhotoAmt']
train_full.drop(toDrop, axis=1, inplace=True)

cat_cols = ['Type', 'Name','Breed1', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength',
            'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State', 'Description'] + \
           list(train_full.columns[train_full.columns.str.contains('label_description')])
train_full[cat_cols] = train_full[cat_cols].apply(lambda x: x.astype('category'))

#Features to keep
features1 = ['Type','Name','Name_length','NameSent','Age','Breed1','Gender','Color1','Color2','Color3','MaturitySize',
             'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
             'VideoAmt', 'Description','Description_length','LexicalDensity',
             'doc_sent_mag', 'doc_sent_score', 'PhotoAmtGood','PhotoAmtFrac','AdoptionSpeed']
features2 = features1 + ['vertex_x-1', 'vertex_yvertex_y-1', 'bounding_confidence-1',
        'bounding_importance-1', 'dominant_blue-1', 'dominant_green-1',
        'dominant_red-1', 'dominant_pixel_frac-1', 'dominant_score-1',
        'label_description-1', 'label_score-1']
features3 = features2 + ['vertex_x-2', 'vertex_yvertex_y-2', 'bounding_confidence-2',
        'bounding_importance-2', 'dominant_blue-2', 'dominant_green-2',
        'dominant_red-2', 'dominant_pixel_frac-2', 'dominant_score-2',
        'label_description-2', 'label_score-2']
features4 = features3 + ['vertex_x-3', 'vertex_yvertex_y-3', 'bounding_confidence-3',
        'bounding_importance-3', 'dominant_blue-3', 'dominant_green-3',
        'dominant_red-3', 'dominant_pixel_frac-3', 'dominant_score-3',
        'label_description-3', 'label_score-3']
features5 = features4 + ['vertex_x-4', 'vertex_yvertex_y-4', 'bounding_confidence-4',
        'bounding_importance-4', 'dominant_blue-4', 'dominant_green-4',
        'dominant_red-4', 'dominant_pixel_frac-4', 'dominant_score-4',
        'label_description-4', 'label_score-4']
features6 = features5 + ['vertex_x-5', 'vertex_yvertex_y-5', 'bounding_confidence-5',
        'bounding_importance-5', 'dominant_blue-5', 'dominant_green-5',
        'dominant_red-5', 'dominant_pixel_frac-5', 'dominant_score-5',
        'label_description-5', 'label_score-5']
features7 = train_full.columns.values
features = []
features.append(features1)
features.append(features2)
features.append(features3)
features.append(features4)
features.append(features5)
features.append(features6)
features.append(features7)

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = qk.quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

#GridSearch
param_dist = {'depth': [4, 7, 10],
          'learning_rate' : [0.03, 0.1, 0.15],
         'l2_leaf_reg': [1,4,9],
         'iterations': [500]}
model = CatBoostRegressor()
my_scorer = make_scorer(qk.quadratic_weighted_kappa, greater_is_better=True)
grid_search = GridSearchCV(model, param_grid=param_dist, cv = 5, verbose = 1, n_jobs=2, scoring=my_scorer)
grid_search.fit(train_full.drop('AdoptionSpeed',axis=1),train_full['AdoptionSpeed'])
grid_search.best_params_

#Optimize CatBoost
params = {
        'loss_function': 'Logloss',
        'max_depth': 7,
        'learning_rate': 0.01,
        'task_type': "GPU",
        'random_strength': 1.5,
        'one_hot_max_size': 30,
        'reg_lambda': 6,
        'od_type': 'Iter',
        'fold_len_multiplier': 2,
        'subsample': 0.85,
        'fold_permutation_block_size': 1,
        'bootstrap_type': 'Bernoulli',
        'random_seed': 17,
        'num_boost_round': 1000}

# Additional parameters:
early_stop = 500
verbose_eval = 100
n_splits = 5

kfold = StratifiedKFold(n_splits=n_splits, random_state=11, shuffle=True)

n = 1
for feature in features:
    train=train_full.loc[:,feature]

    # get the categorical features
    cat_feature_names = train.dtypes[train.dtypes == "category"]
    cat_features = [train.columns.get_loc(c) for c in train.columns if c in cat_feature_names]

    qwk_scores = []
    feature_importance_df = pd.DataFrame()
    i=1
    for train_index, valid_index in kfold.split(train, train['AdoptionSpeed'].values):
            X_tr = train.iloc[train_index, :]
            X_val = train.iloc[valid_index, :]

            y_tr = X_tr['AdoptionSpeed'].values
            X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

            y_val = X_val['AdoptionSpeed'].values
            X_val = X_val.drop(['AdoptionSpeed'], axis=1)

            watchlist = (X_val, y_val)

            model = CatBoostClassifier(cat_features=list(cat_features), **params)
            model.fit(X_tr, y_tr, eval_set=watchlist, verbose=verbose_eval,early_stopping_rounds=early_stop)
            val_pred = model.predict(X_val)
            val_predictions = []
            for x in val_pred:
                val_predictions.append(x)

            # optR = OptimizedRounder()
            # optR.fit(val_pred, y_val)
            # coefficients = optR.coefficients()
            # val_predictions = optR.predict(val_pred, coefficients)
            qwk = qk.quadratic_weighted_kappa(y_val, val_predictions)
            qwk_scores.append(qwk)
            importances = model.get_feature_importance()
            fold_importance_df = pd.DataFrame()
            fold_importance_df['feature'] = train.drop(['AdoptionSpeed'],axis=1).columns.values
            fold_importance_df['importance'] = importances
            fold_importance_df['fold'] = i
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            i+=1


    print('Results for CatBoost regression - model',n)
    print('QWK scores : {}'.format(qwk_scores))
    print('mean QWK score : {}'.format(np.mean(qwk_scores)))
    print(feature_importance_df.groupby('feature')['feature', 'importance'].mean().reset_index().sort_values('importance', ascending=False).head(50))
    n+=1


