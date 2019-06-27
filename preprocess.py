import pandas as pd
import numpy as np
import json
import os

from textblob import TextBlob

def preprocess (whichData, sent=True, meta=True):
    path = "../Data/"
    colors = pd.read_csv(os.path.join(path,'color_labels.csv'))
    breeds = pd.read_csv(os.path.join(path,'breed_labels.csv'))
    if whichData != 'train' and whichData != 'test':
        return []
    dataset = pd.read_csv(os.path.join(path,whichData+'/'+whichData+'.csv'))
    dataset_probs = pd.read_csv(os.path.join(path,whichData+'/'+whichData+'_probs.csv')) #Label 0 dog, label 1 cat.
    sentiment_dir = os.path.join(path,whichData+'_sentiment')
    metadata_dir = os.path.join(path,whichData+'_metadata')
    '''
        #Names:
        - Names are in fact, in some cases, a short description of the pet.
            Those profiles with information in the name feature may have a faster adoption speed.
        - Some names have a code starting with PF
        - Names with URGENT included can be helpful
        - Names with Puppy, Kitty or any additional description can help
        - Sentiment analysis may help. I presume that those with negatives name description will adopt faster.
    '''

    cat_cols = ['Type','Name','Breed1','Gender','Color1','Color2','Color3','MaturitySize','FurLength',
                'Vaccinated','Dewormed','Sterilized','Health','State', 'RescuerID','Description','PetID']
    #Names
    dataset.insert(dataset.columns.get_loc('Name')+1, 'Name_length', 0)  # Add feature Name_length with default value 0
    dataset['Name_length'] = np.where(dataset['Name'].str.split().str.len() > 0, dataset['Name'].str.split().str.len(), 0) #Add Name_length (in words)
    dataset['Name'].fillna('unnamed', inplace=True)  # Fill NaN with unnamed

    #Sentiment on Name
    dataset.insert(dataset.columns.get_loc('Name_length') + 1, 'NameSent',0)  # Add feature Name_length with default value 0
    dataset['NameSent'] = dataset['Name'].astype(str).apply(lambda x: TextBlob(x).sentiment[0])

    dataset[['Name', 'NameSent']].head()
    # Assign dummy values to Name -> 4=cat, 3=kitty, 2=dog, 1=puppy, 0=other names,  -1=unnamed or no name
    dataset['Name'] = np.select([dataset['Name'].str.contains('puppy', case=False),
                                 dataset['Name'].str.contains('dog', case=False),
                                 dataset['Name'].str.contains('kitty', case=False),
                                 dataset['Name'].str.contains('cat', case=False),
                                dataset['Name'].str.contains('no name|no-name|not name|yet|unnamed',case=False)],
                                [1, 2, 3, 4, -1],0)

    #Description
    dataset.insert(dataset.columns.get_loc('Description')+1, 'Description_length', 0)  # Add feature Description_length with default value 0
    dataset['Description_length'] = np.where(dataset['Description'].str.split().str.len() > 0, dataset['Description'].str.split().str.len(), 0) #Add Description_length (in words)

    dataset.insert(dataset.columns.get_loc('Description_length') + 1, 'LexicalDensity',0)  # Add feature Name_length with default value 0
    dataset['LexicalDensity'] = dataset['Description'].str.lower().str.split().apply(lambda x: np.unique(x)).str.len() / dataset['Description_length']
    dataset['LexicalDensity'].replace([np.inf, -np.inf],0, inplace=True)

    dataset['Description'].fillna('nothing', inplace=True)  # Fill NaN with nothing

    # Assign dummy values to Description -> 4=cat, 3=kitty, 2=dog, 1=puppy, 0=other names,  -1=unnamed or no name
    dataset['Description'] = np.select([dataset['Description'].str.contains('puppy', case=False),
                                        dataset['Description'].str.contains('dog', case=False),
                                        dataset['Description'].str.contains('kitty', case=False),
                                        dataset['Description'].str.contains('cat', case=False),
                                        dataset['Description'].str.contains('nothing', case=False)],
                                       [1, 2, 3, 4, -1], 0)
    # Breed2 = 0 if Breed1 == Breed2
    dataset['Breed2'] = np.where((dataset['Breed1'] == dataset['Breed2']), 0, dataset['Breed2'])
    # Breed1 = Breed2 and set Breed2 = 0 if Breed1 = 0 and Breed2 != 0
    zeroBreed1 = (dataset['Breed1'] == 0)
    dataset.loc[zeroBreed1, 'Breed1'] = dataset['Breed2']
    dataset.loc[zeroBreed1, 'Breed2'] = 0
    #If Breed1 and Breed 2 change Breed1 for 307 (Mixed Breed)
    dataset.loc[dataset['Breed2'] != 0,'Breed1'] = 307
    #Drop Breed2
    dataset.drop('Breed2', axis=1, inplace=True)
    #Set incorrect breed types to 307
    datasetBreed = dataset.merge(breeds, how='left', left_on='Breed1', right_on='BreedID', suffixes=('', '_br'))
    dataset.loc[datasetBreed.Type != datasetBreed.Type_br,'Breed1'] = 307
    del datasetBreed

    # RescuerID doesn't help because there is no overlap between training and testing
    '''
    #Label encoding the RescuerID feature
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
    dataset_id = dataset['PetID']
    # Add sentiment data
    if sent:
        doc_sent_mag = []
        doc_sent_score = []
        for pet in dataset_id:
            try:
                with open(os.path.join(sentiment_dir, pet + '.json'), 'r', encoding="utf8") as f:
                    sentiment = json.load(f)
                doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
                doc_sent_score.append(sentiment['documentSentiment']['score'])
            except FileNotFoundError:
                doc_sent_mag.append(-1)
                doc_sent_score.append(-1)

        dataset.loc[:, 'doc_sent_mag'] = doc_sent_mag
        dataset.loc[:, 'doc_sent_score'] = doc_sent_score

    # Add image metadata
    if meta:
        dataset.loc[:,'PhotoAmtGood'] = 0
        image_id = dataset_probs['ImageID']
        im_cnt = 0
        for n,im in enumerate(image_id):
            if dataset_probs.loc[n,'Label']==0:
                dataset_probs.loc[n,'Prob'] = 1-float(dataset_probs.loc[n,'Prob'])
            if dataset_probs.loc[n,'Label']==dataset_probs.loc[n,'Pred'] and dataset_probs.loc[n,'Prob'] >= 0.99:
                im_cnt+=1
                with open(os.path.join(metadata_dir, im + '.json'), 'r', encoding="utf8") as f:
                    data = json.load(f)
                vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
                vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
                bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
                bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
                if bool(data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']):
                    dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
                    dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
                    dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
                else:
                    dominant_blue=-1
                    dominant_green=-1
                    dominant_red=-1
                dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
                dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
                dataset.loc[dataset['PetID'] == im.split('-')[0], 'PhotoAmtGood'] += 1
                if data.get('labelAnnotations'):
                    label_description = data['labelAnnotations'][0]['description']
                    label_score = data['labelAnnotations'][0]['score']
                else:
                    label_description = 'nothing'
                    label_score = -1
            else:
                vertex_x = -1
                vertex_y = -1
                bounding_confidence = -1
                bounding_importance_frac = -1
                dominant_blue = -1
                dominant_green = -1
                dominant_red = -1
                dominant_pixel_frac = -1
                dominant_score = -1
                label_description = 'nothing'
                label_score = -1
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'vertex_x-' + im.split('-')[1]] = vertex_x
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'vertex_y''vertex_y-' + im.split('-')[1]] = vertex_y
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'bounding_confidence-' + im.split('-')[1]] = bounding_confidence
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'bounding_importance-' + im.split('-')[1]] = bounding_importance_frac
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'dominant_blue-' + im.split('-')[1]] = dominant_blue
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'dominant_green-' + im.split('-')[1]] = dominant_green
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'dominant_red-' + im.split('-')[1]] = dominant_red
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'dominant_pixel_frac-' + im.split('-')[1]] = dominant_pixel_frac
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'dominant_score-' + im.split('-')[1]] = dominant_score
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'label_description-' + im.split('-')[1]] = label_description
            dataset.loc[dataset['PetID'] == im.split('-')[0], 'label_score-' + im.split('-')[1]] = label_score
        dataset['Name'].fillna('unnamed', inplace=True)  # Fill NaN with unnamed
        for col,descN in enumerate(pd.Series(list(dataset)).str.contains('label_description')):
            if descN:
                dataset.iloc[:, col].fillna('nothing', inplace=True) #Fill Na label_descriptions with 'nothing'
                # Assign dummy values to label_description -> 4=cat, 3=kitty, 2=dog, 1=puppy, 0=other names,  -1=unnamed or no name
                dataset.iloc[:,col]= np.select([dataset.iloc[:, col].str.contains('puppy', case=False),
                                             dataset.iloc[:, col].str.contains('dog', case=False),
                                             dataset.iloc[:, col].str.contains('kitty', case=False),
                                             dataset.iloc[:, col].str.contains('cat', case=False),
                                             dataset.iloc[:, col].str.contains('nothing', case=False)],
                                            [1, 2, 3, 4, -1], 0)
                cat_cols.append(list(dataset)[col])
            else:
                dataset.iloc[:, col].fillna(-1, inplace=True)  # Fill other Na metadata with -1
        print("Number of correctly classified images for " + whichData + ": %s" % im_cnt)
        dataset.loc[:,'PhotoAmtFrac'] = dataset['PhotoAmtGood']/dataset['PhotoAmt']
        dataset['PhotoAmtFrac'].fillna(-1, inplace=True) #These are pets with no photos PhotoAmt=0

    dataset[cat_cols] = dataset[cat_cols].apply(lambda x: x.astype('category'))
    # Breeds, check if they have only Breed1 and is mixed or mixed is written in name or description
    # FurLength, check if it is written in name or description. If yes, correct the FurLength accordingly
    return dataset