import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def get_labels(df):
    labels = df['poor'].astype(int)
    df.drop('poor', axis=1, inplace=True)
    return df, labels

def convert_string_to_categorical(df):
    try:
        df.drop(['id', 'iid', 'country'], axis=1, inplace=True) #drop id and country as they are irrelevant features
    except:
        df.drop(['id', 'country'], axis=1, inplace=True)
    all_features = list(df.columns.values)
    object_features = []

    for feature in all_features:
        if df[feature].dtype == 'object':
            object_features.append(feature)

    for feature in object_features:
        df[feature] = df[feature].astype('category')
        df[feature] = df[feature].cat.codes
    # dict(enumerate(df['KAJOWiiw'].cat.categories)) #to learn mapping between category and string
    return df

def standardize(df):
    standardizer = StandardScaler().fit(df)
    standardized_data = standardizer.transform(df)
    standardized_data = pd.DataFrame(standardized_data)
    standardized_data.columns = df.columns
    return standardized_data

def rescale(df, type='minmax'):
    if type == 'minmax':
        normalizer = MinMaxScaler()
    elif type == 'robust':
        normalizer = RobustScaler()
    rescaled_data = normalizer.fit_transform(df)
    rescaled_data = pd.DataFrame(rescaled_data)
    rescaled_data.columns = df.columns
    return rescaled_data

def missing_data(df, missing_features, remove_features):

    if remove_features!=[]:
        df.drop(remove_features, axis=1, inplace=True)

    for feature in missing_features:
        df[feature].fillna(df[feature].mean(), inplace=True)

    return df

def check_class_imbalance(labels):
    ones = (labels==1).sum()
    zeros = (labels==0).sum()
    percent_ones = ones/labels.count()
    percent_zeros = zeros/labels.count()
    print('Number of ones:', ones,',','Percentage:',percent_ones)
    print('Number of zeros:', zeros, ',', 'Percentage:', percent_zeros)

def treat_imbalance(df, labels):
    df['labels'] = labels

    ada = SMOTE(k_neighbors=4)
    resampled_data, resampled_labels = ada.fit_sample(df.drop('labels', axis=1), labels)

    resampled_data = pd.DataFrame(resampled_data)
    resampled_labels = pd.DataFrame(resampled_labels)
    resampled_data.columns = df.drop('labels', axis=1).columns
    resampled_data['labels'] = resampled_labels
    resampled_data = resampled_data.sample(frac=1).reset_index(drop=True)
    resampled_labels = resampled_data['labels']
    resampled_data.drop('labels', axis=1, inplace=True)

    return resampled_data, resampled_labels

def select_features(k, train, labels_tr, val):
    features = SelectKBest(score_func=f_classif, k=k)
    new_train = features.fit_transform(train, labels_tr)
    new_val = features.transform(val)
    return new_train, new_val, features
