import pandas as pd

def load_data(fileName):
    base = 'F:\\Python\Predicting Poverty\\train\\'
    if fileName == 'A_hhold_train':
        df = pd.read_csv(base+'A_hhold_train.csv')
    elif fileName == 'B_hhold_train':
        df = pd.read_csv(base+'B_hhold_train.csv')
    elif fileName == 'C_hhold_train':
        df = pd.read_csv(base+'C_hhold_train.csv')
    elif fileName == 'A_indiv_train':
        df = pd.read_csv(base+'A_indiv_train.csv')
    elif fileName == 'B_indiv_train':
        df = pd.read_csv(base+'B_indiv_train.csv')
    elif fileName == 'C_indiv_train':
        df = pd.read_csv(base+'C_indiv_train.csv')

    base = 'F:\\Python\Predicting Poverty\\test\\'
    if fileName == 'A_hhold_test':
        df = pd.read_csv(base+'A_hhold_test.csv')
    elif fileName == 'B_hhold_test':
        df = pd.read_csv(base+'B_hhold_test.csv')
    elif fileName == 'C_hhold_test':
        df = pd.read_csv(base+'C_hhold_test.csv')
    elif fileName == 'A_indiv_test':
        df = pd.read_csv(base+'A_indiv_test.csv')
    elif fileName == 'B_indiv_test':
        df = pd.read_csv(base+'B_indiv_test.csv')
    elif fileName == 'C_indiv_test':
        df = pd.read_csv(base+'C_indiv_test.csv')
    return df

def save_df_to_csv(df, filename):
    df.to_csv(filename, encoding='utf-8', index=False, sep=',')


def split_data(df, labels, train_percent):
    df['labels'] = labels
    poor = df[df['labels'] == 1]
    not_poor = df[df['labels'] == 0]

    train = poor[:round(train_percent * poor.shape[0])]
    val = poor[round(train_percent * poor.shape[0]):]

    train = train.append(not_poor[:round(train_percent * not_poor.shape[0])], ignore_index=True)
    val = val.append(not_poor[round(train_percent * not_poor.shape[0]):], ignore_index=True)

    train = train.sample(frac=1).reset_index(drop=True)
    val = val.sample(frac=1).reset_index(drop=True)
    labels_tr = train['labels']
    labels_val = val['labels']
    return train.drop('labels', axis=1), labels_tr, val.drop('labels', axis=1), labels_val