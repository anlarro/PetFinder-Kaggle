import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics.scorer import make_scorer

#read data
train = pd.read_csv("../Data/train/train.csv")
test = pd.read_csv("../Data/test/test.csv")
colors = pd.read_csv("../Data/color_labels.csv")
breeds = pd.read_csv("../Data/breed_labels.csv")

#Let's check the most popular names
train['Name'].value_counts().head(50)
#Apparently those Names with puppy or kitty words are adopted faster. When people look for puppy with the search option of petfinder,
# these profiles appear first
train['AdoptionSpeed'].where(train['Name'].str.contains('pup',case=False)).value_counts().sort_index().plot('barh', color='blue')
plt.title('AdoptionSpeed for names containing puppy/puppies')

train['AdoptionSpeed'].where(train['Name'].str.contains('kit',case=False)).value_counts().sort_index().plot('barh', color='red')
plt.title('AdoptionSpeed for names containing kitty/kitties')


#Declare functions
def preprocess (dataset):

    '''
        #Names:
        - Names are in fact, in some cases, a short description of the pet.
            Those profiles with information in the name feature may have a faster adoption speed.
        - Some names have a code starting with PF
        - Names with URGENT included can be helpful
        - Names with Puppy, Kitty or any aditional description can help
        - Sentiment analysis may help. I presume that those with negatives name description will adopt faster.
    '''
    '''
    This doesn't seem to help, better to use pup/kit as name1, other as name2 and unnamed as name3
    #Fetch popular names from public database
    s=requests.get("http://rochester.nydatabases.com/sites/default/files/popular%20names_0.csv").content
    popular_dogNames=pd.read_csv(io.StringIO(s.decode('utf-8')))
    popular_dogNames = popular_dogNames[popular_dogNames.Number > 1] #Keep popular names with Number > 1
    popular_dogNames.Name = popular_dogNames.Name.str.lower() #lowercase popular names

    train['Name'].fillna('unnamed',inplace=True) #Assign 0 to NaN values
    train.Name = train.Name.str.lower()   #lowercase all names
    train.Name.where(train.Name.isin(popular_dogNames.Name),other='unnamed',inplace=True)
    ##Some cases still have no name or girl/boy
    '''
    #Names
    dataset.insert(dataset.columns.get_loc('Name')+1, 'Name_length', 0)  # Add feature Name_length with default value 0
    dataset['Name_length'] = np.where(dataset['Name'].str.split().str.len() > 0, dataset['Name'].str.split().str.len(), 0) #Add Name_length (in words)
    dataset['Name'].fillna('unnamed', inplace=True)  # Fill NaN with unnamed

    # Assign dummy values to Name -> 2=contains pup or kit, 1=other names,  0=unnamed or no name
    dataset['Name'] = np.select([dataset['Name'].str.contains('pup|kit', case=False),
                                 dataset['Name'].str.contains('no name|no-name|not name|yet|unname', case=False)], [2, 0],1)

    #Description
    dataset.insert(dataset.columns.get_loc('Description')+1, 'Description_length', 0)  # Add feature Description_length with default value 0
    dataset['Description_length'] = np.where(dataset['Description'].str.split().str.len() > 0, dataset['Description'].str.split().str.len(), 0) #Add Description_length (in words)
    dataset['Description'].fillna('nothing', inplace=True)  # Fill NaN with nothing

    # Assign dummy values to Description -> 2=contains pup or kit, 1=other,  0=nothing
    dataset['Description'] = np.select([dataset['Description'].str.contains('pup|kit', case=False),
                                 dataset['Description'].str.contains('nothing', case=False)], [2, 0],1)

    # Age, delete outliers (pet over 15years
    # Breeds, check if they have only Breed 1 and is mixed or mixed is written in name or description
    # FurLength, check if it is written in name or description. If yes, correct the FurLength accordingly
    return dataset

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

#Preprocess data
train=preprocess(train.copy())
test=preprocess(test.copy())

labels=train['AdoptionSpeed']
train=train.drop(['AdoptionSpeed','PetID','RescuerID'],axis=1)
test=test.drop(['PetID','RescuerID'],axis=1)

#RescuerID doesn't help because there is no overlap between training and testing
'''
#OneHot encoding the RescuerID feature
encoder = LabelEncoder().fit(train.RescuerID)
encoder2 = LabelEncoder().fit(test.RescuerID)
rid_train=dict(zip(list(encoder.classes_),encoder.transform(train.RescuerID)))
rid_test=dict(zip(list(encoder2.classes_),max(rid_train.values()) + 1 + encoder2.transform(test.RescuerID)))
#Check if RescuerID in test is also present in train, if not leave the new encoded value assigned in the previoues line
for key in rid_test.items() & rid_train.items():
    if rid_test[key]==rid_train[key]:
        rid_test[key]=rid_train[key]
train['RescuerID'].replace(rid_train, inplace=True)
test['RescuerID'].replace(rid_test, inplace=True)
'''

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
rf_random=RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 300, cv = 10, verbose = 1, random_state=11,n_jobs=4,scoring=my_scorer)
rf_random.fit(train,labels)
rf_random.best_params_

#Best parameters
rf_paramsKappa_{
 n_estimators= 400,
 min_samples_split= 2,
 min_samples_leaf= 2,
 max_features= 'sqrt',
 max_depth= 30,
 bootstrap= False}

rf=RandomForestClassifier(n_estimators= 400,min_samples_split= 2,min_samples_leaf= 2,max_features= 'sqrt',max_depth= 30,bootstrap= False,verbose=0,random_state=11)
rf.fit(train,labels)
predictions=rf.predict(test)

np.unique(predictions, return_counts=True)
feature_importances = pd.DataFrame(rf.feature_importances_,index = train.columns,columns=['importance']).sort_values('importance',ascending=False)

#Add json metadata
#Optimize random forest and submit
#Optimize xgboost and submit
#Merge breeds with the same name. Verify mix in name or description and in breed1 and breed2
#Delete outliers in age and other possible features
#Optimize xgboost and submit
#Fit random forest with standard params and select features (5,10,15,20)
#Optimize xgboost with above selected features
#Run nnet for profile photo and stack with xgboost





