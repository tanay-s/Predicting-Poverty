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

def handle_missing_data(df, missing_features = ['BXOWgPgL', 'McFBIGsm', 'BRzuVmyf', 'OSmfjCbE'],
                        remove_features = ['umkFMfvA','FGWqGkmD','IrxBnWxE','dnlnKrAg','aAufyreG']):
    if remove_features!=[]:
        df.drop(remove_features, axis=1, inplace=True)
    # df = standardize(df)
    df = df.sample(frac=1).reset_index(drop=True)
    complete_rows = df.dropna()
    missing_rows = df[df.isnull().any(axis=1)]
    data = complete_rows.drop(missing_features, axis=1)
    #standardise df for test data
    id = data['id']
    id = id.reset_index(drop=True)
    data.drop('id', axis=1, inplace=True)
    data = standardize(data)
    data = pd.DataFrame(data)
    data['id'] = id
    data.columns = complete_rows.drop(missing_features, axis=1).columns
    data['id'] = id
    validation_index = data.shape[0] - round(0.2*data.shape[0])
    training_data = data[:validation_index]
    validation_data = data[validation_index:]
    # temp_data = missing_rows.drop(missing_features, axis=1)
    for feature in missing_features:
        labels = complete_rows[feature]
        labels_training = labels[:validation_index]
        labels_validation = labels[validation_index:]
        temp_data = missing_rows[missing_rows[feature].isnull()]
        test_data = temp_data.drop(missing_features, axis=1)

        regressor = Lasso(alpha=1)
        regressor.fit(training_data.drop('id', axis=1), labels_training)
        predictions_val = regressor.predict(validation_data.drop('id',axis=1))

        val_loss = mean_absolute_error(labels_validation, predictions_val)
        print("loss for",feature,":", val_loss)

        # predictions = regressor.predict(test_data.drop('id',axis=1))

def find_missing_values(df):
    def num_missing(x):
        return sum(x.isnull())

    pd.set_option("display.max_rows", 999) #uncomment to display all columns
    missing_values_per_column = df.apply(num_missing, axis=0)
    print (missing_values_per_column)
    print('Number of missing values:', sum(missing_values_per_column))

def missing_data(df, missing_features = ['BXOWgPgL', 'McFBIGsm', 'BRzuVmyf', 'OSmfjCbE'],
                remove_features = ['umkFMfvA','FGWqGkmD','IrxBnWxE','dnlnKrAg','aAufyreG']):

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
