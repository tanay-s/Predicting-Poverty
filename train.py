from data_preprocessing import *
from utils import *
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
import pandas as pd

def train(file, rescale=False, treat_imb=False, k=0):
    data = load_data(file)
    data, labels = get_labels(data)
    data = convert_string_to_categorical(data)
    if file == 'B_hhold_train':
        data = missing_data(data, missing_features = ['BXOWgPgL', 'McFBIGsm', 'BRzuVmyf', 'OSmfjCbE'],
                            remove_features=['umkFMfvA', 'FGWqGkmD', 'IrxBnWxE', 'dnlnKrAg', 'aAufyreG'])

    elif file == 'A_indiv_train':
        data = missing_data(data, missing_features=['OdXpbPGJ'], remove_features=[])

    elif file == 'B_indiv_train':
        data = missing_data(data, missing_features=['mAeaImix', 'jzBRbsEG', 'TZDgOhYY', 'esHWAAyG','TJGiunYp'],
                      remove_features=['NfpXxGQk', 'tzYvQeOb', 'HZqPmvkr', 'nxAFXxLQ', 'fyfDnyQk',
                            'iZhWxnWa', 'AJgudnHB', 'hdDTwJhQ', 'uDmhgsaQ', 'sIiSADFG',
                            'DSttkpSI','WqEZQuJP', 'CLTXEwmz', 'sWElQwuC','ETgxnJOM','DtcKwIEv',
                            'DYgxQeEi', 'jfsTwowc','MGfpfHam','WmKLEUcd', 'unRAgFtX', 'qlLzyqpP',
                            'BoxViLPz'])

    else:
        find_missing_values(data)

    if rescale:
        data = rescale(data)

    train, labels_tr, val, labels_val = split_data(data, labels, 0.9)

    if treat_imb:
        train, labels_tr = treat_imbalance(train, labels_tr)

    if k > 0:
        train, val, features = select_features(k, train, labels_tr, val)
    else:
        features = None

    xgb = XGBClassifier(n_estimators=100)
    xgb.fit(train, labels_tr)
    preds = xgb.predict_proba(val)
    loss = log_loss(labels_val, preds)
    print('Loss:', loss)
    return xgb, features

def infer(file, estimator, rescale=False, features=False):
    data = load_data(file)
    data = convert_string_to_categorical(data)
    if file == 'B_hhold_test':
        data = missing_data(data, missing_features = ['BXOWgPgL', 'McFBIGsm', 'BRzuVmyf', 'OSmfjCbE'],
                            remove_features=['umkFMfvA', 'FGWqGkmD', 'IrxBnWxE', 'dnlnKrAg', 'aAufyreG'])

    elif file == 'A_indiv_test':
        data = missing_data(data, missing_features=['OdXpbPGJ'], remove_features=[])

    elif file == 'B_indiv_test':
        data = missing_data(data, missing_features=['mAeaImix', 'jzBRbsEG', 'TZDgOhYY', 'esHWAAyG','TJGiunYp'],
                      remove_features=['NfpXxGQk', 'tzYvQeOb', 'HZqPmvkr', 'nxAFXxLQ', 'fyfDnyQk',
                            'iZhWxnWa', 'AJgudnHB', 'hdDTwJhQ', 'uDmhgsaQ', 'sIiSADFG',
                            'DSttkpSI','WqEZQuJP', 'CLTXEwmz', 'sWElQwuC','ETgxnJOM','DtcKwIEv',
                            'DYgxQeEi', 'jfsTwowc','MGfpfHam','WmKLEUcd', 'unRAgFtX', 'qlLzyqpP',
                            'BoxViLPz'])

    else:
        find_missing_values(data)

    if rescale:
        data = rescale(data)

    if features:
        data = features.transform(data)

    predictions = estimator.predict_proba(data)
    return predictions

def output_format(file, predictions):
    data = load_data(file)
    output = pd.DataFrame()
    output['id'] = data['id']
    output['country'] = data['country']
    output['poor'] = predictions[:,1]
    return output


if __name__ == '__main__':
    estimator, features = train('A_hhold_train', treat_imb=True)
    preds = infer('A_hhold_test', estimator)
    out1 = output_format('A_hhold_test', preds)

    estimator, features = train('B_hhold_train', treat_imb=True)
    preds = infer('B_hhold_test', estimator)
    out2 = output_format('B_hhold_test', preds)

    estimator, features = train('C_hhold_train', treat_imb=True)
    preds = infer('C_hhold_test', estimator)
    out3 = output_format('C_hhold_test', preds)

    output = pd.concat([out1, out2, out3], ignore_index=True)
    save_df_to_csv(output, 'output.csv')


