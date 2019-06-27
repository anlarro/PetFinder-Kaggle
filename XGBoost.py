import pandas as pd
import numpy as np
import scipy as sp
from preprocess import preprocess
import QuadraticKappa as qk
from functools import partial

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer

import category_encoders

#Preprocess data
# train_full=preprocess('train')
# test_full=preprocess('test')
# train_full.to_csv('../Data/train/trainProc.csv',index=False)
# test_full.to_csv('../Data/test/testProc.csv',index=False)

#Loas Preprocessed data
train_full = pd.read_csv('../Data/train/trainProc_fineTune.csv')
test_full = pd.read_csv('../Data/test/testProc_fineTune.csv')
#We drop 'RescuerID' as there is not overlap between train and test
#We drop 'PhotoAmt' as we already have 'PhotoAmtGood' and 'PhotoAmtFrac=PhotoAmtGood/PhotoAmt'
toDrop = ['PetID','RescuerID','PhotoAmt']
train_full.drop(toDrop, axis=1, inplace=True)

cat_cols = ['Type', 'Name','Breed1', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength',
            'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State', 'Description'] + \
           list(train_full.columns[train_full.columns.str.contains('label_description')])
train_full[cat_cols] = train_full[cat_cols].apply(lambda x: x.astype('category'))
test_full[cat_cols] = test_full[cat_cols].apply(lambda x: x.astype('category'))

#Features to keep
# features1 = ['Type','Age','Breed1','Gender','Quantity', 'Description_length',
#              'doc_sent_mag', 'doc_sent_score', 'PhotoAmtGood','PhotoAmtFrac','AdoptionSpeed']
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

# Additional parameters:
n_splits = 5

kfold = StratifiedKFold(n_splits=n_splits, random_state=11, shuffle=True)

results = []
n = 1
for feature in features:
    train=train_full.loc[:,feature]

    qwk_scores = []
    feature_importance_df = pd.DataFrame()
    i=1
    for train_index, valid_index in kfold.split(train, train['AdoptionSpeed'].values):
#######################################################################################################
            #Profiles with Good images
            X_tr = train.iloc[train_index, :]
            X_tr = X_tr[X_tr.PhotoAmtGood != 0].reset_index(drop=True)  # We keep rows with Good images

            X_val = train.iloc[valid_index, :]
            X_val = X_val[X_val.PhotoAmtGood != 0].reset_index(drop=True)  # We keep rows with Good images

            y_tr = X_tr['AdoptionSpeed']
            X_tr.drop('AdoptionSpeed', axis=1,inplace=True)

            y_val1 = X_val['AdoptionSpeed']
            X_val.drop('AdoptionSpeed', axis=1,inplace=True)

            # get the categorical features
            cat_cols = list(X_tr.columns[X_tr.dtypes == "category"])
            te = category_encoders.TargetEncoder(cols=cat_cols, smoothing=1)
            te = te.fit(X_tr,y_tr)
            X_tr = te.transform(X_tr)
            X_val = te.transform(X_val)

            model = XGBRegressor(eval_metric='rmse', early_stopping_rounds=500,
                                    objective='reg:linear', silent=True,
                                    subsample=1.0, reg_lambda=100.0, reg_alpha=0.01, n_estimators=5000,
                                    min_child_weight=5,
                                    max_depth=15, learning_rate=0.01, gamma=0.25, colsample_bytree=1.0,
                                    colsample_bylevel=0.6,
                                    verbose=2, random_state=11)
            model.fit(X_tr, y_tr)
            val_pred = model.predict(X_val)
            tr_pred = model.predict(X_tr)

            # val_predictions = []
            # for x in val_pred:
            #     val_predictions.append(np.argmax(x))

            optR = OptimizedRounder()
            optR.fit(tr_pred, y_tr)
            coefficients = optR.coefficients()
            val_predictions1 = optR.predict(val_pred, coefficients)

            #Profiles without Good images
            X_tr = train.iloc[train_index, :]
            X_tr = X_tr[X_tr.PhotoAmtGood == 0].reset_index(drop=True)  # We keep rows with no Good images

            X_val = train.iloc[valid_index, :]
            X_val = X_val[X_val.PhotoAmtGood == 0].reset_index(drop=True)  # We keep rows with no Good images

            y_tr = X_tr['AdoptionSpeed']
            X_tr.drop('AdoptionSpeed', axis=1,inplace=True)

            y_val2 = X_val['AdoptionSpeed']
            X_val.drop('AdoptionSpeed', axis=1,inplace=True)

            # get the categorical features
            cat_cols = list(X_tr.columns[X_tr.dtypes == "category"])
            te = category_encoders.TargetEncoder(cols=cat_cols, smoothing=1)
            X_tr = te.fit_transform(X_tr,y_tr)
            X_val = te.transform(X_val)

            model = XGBRegressor(eval_metric='rmse', early_stopping_rounds=500,
                     objective='reg:linear', silent=True,
                     subsample=1.0, reg_lambda=100.0, reg_alpha=0.01, n_estimators=5000,
                     min_child_weight=5,
                     max_depth=15, learning_rate=0.01, gamma=0.25, colsample_bytree=1.0,
                     colsample_bylevel=0.6,
                     verbose=2, random_state=11)
            model.fit(X_tr, y_tr)
            val_pred = model.predict(X_val)
            tr_pred = model.predict(X_tr)
            # val_predictions = []
            # for x in val_pred:
            #     val_predictions.append(np.argmax(x))

            optR = OptimizedRounder()
            optR.fit(tr_pred, y_tr)
            coefficients = optR.coefficients()
            val_predictions2 = optR.predict(val_pred, coefficients)

#########OVERALL RESULTS###########
            qwk = qk.quadratic_weighted_kappa(np.concatenate((y_val1,y_val2),axis=0),
                                              np.concatenate((val_predictions1, val_predictions2),axis=0))
            qwk_scores.append(qwk)
            importances = model.feature_importances_
            fold_importance_df = pd.DataFrame()
            fold_importance_df['feature'] = train.drop(['AdoptionSpeed'],axis=1).columns.values
            fold_importance_df['importance'] = importances
            fold_importance_df['fold'] = i
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            i+=1

    print('Results for LGB classification - model',n)
    print('QWK scores : {}'.format(qwk_scores))
    print('mean QWK score : {}'.format(np.mean(qwk_scores)))
    results.append(np.mean(qwk_scores))
    print('std QWK score : {}'.format(np.std(qwk_scores)))
    print(feature_importance_df.groupby('feature')['feature', 'importance'].mean().reset_index().sort_values('importance', ascending=False).head(50))
    n+=1

# Predict test set

